# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
"""Import-time provisioning of HYBM AICPU Kernel for installed wheel."""

import errno
import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time


_VERSION_FILENAME = "cann_hybm_kernel_version"
_TAR_FILENAME = "cann-hybm-compat.tar.gz"
_JSON_FILENAME = "libcann_hybm_kernel.json"
_INI_FILENAME = "ascend_package_load.ini"
_WHEEL_MARKER = ".hybm_aicpu_provision_wheel_only"

_KERNEL_REL = "opp/vendors/cust/op_impl/aicpu/kernel"
_CONFIG_REL = "opp/vendors/cust/op_impl/aicpu/config"
_CONF_REL = "conf"
_LOCK_FILENAME = ".cann_hybm_kernel_provision.lock"

_INI_STANDARD_BLOCK = (
    "name:cann-hybm-compat.tar.gz\n"
    "install_path:2\n"
    "optional:true\n"
    "package_path:opp/vendors/cust/op_impl/aicpu/kernel\n"
    "load_as_per_soc:false\n"
)

_LOCK_TIMEOUT_SEC = 120
_BUILD_TIMEOUT_SEC = 120
_STDERR_TRUNC = 4096

_TAR_EXPECTED_FILE = "aicpu_kernels_device/libcann_hybm_kernel.so"
_TAR_EXPECTED_DIR = "aicpu_kernels_device"


class ProvisioningError(Exception):
    """Carries stage, exit code (or None), truncated stderr, temp dir."""

    def __init__(self, stage, exit_code, stderr_truncated, temp_dir):
        parts = ["AICPU kernel provisioning failed at stage '%s'" % stage]
        if exit_code is not None:
            parts.append("exit code %s" % exit_code)
        if stderr_truncated:
            parts.append("stderr: %s" % stderr_truncated)
        if temp_dir:
            parts.append("build dir preserved: %s" % temp_dir)
        super().__init__(", ".join(parts))
        self.stage = stage
        self.exit_code = exit_code
        self.stderr_truncated = stderr_truncated
        self.temp_dir = temp_dir


# ========== helpers ==========


def _get_pkg_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _rand_hex(nbytes=6):
    return os.urandom(nbytes).hex()


def _write_all(fd, data):
    """Write all data to fd; raises OSError on short write."""
    while data:
        written = os.write(fd, data)
        if written <= 0:
            raise OSError("short write: wrote %d of %d bytes" % (written, len(data)))
        data = data[written:]


# ========== anchor + directory traversal with symlink resolution ==========


_MAX_SYMLINK_COUNT = 40


def _open_anchor(ascend_home):
    """Resolve ASCEND_HOME_PATH, return (canonical_path, O_DIRECTORY|O_NOFOLLOW fd)."""
    real = os.path.realpath(ascend_home)
    fd = os.open(real, os.O_DIRECTORY | os.O_NOFOLLOW)
    return real, fd


def _create_dir(p, cur):
    """Create dir p relative to cur fd; handle concurrent create.

    Returns child fd.  Only fchmod 0750 when this call created the
    directory.  Does not modify umask.  Closes child on fchmod failure.
    """
    try:
        os.mkdir(p, mode=0o700, dir_fd=cur)
        created = True
    except FileExistsError:
        created = False
    child = os.open(p, os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur)
    if created:
        try:
            os.fchmod(child, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
        except BaseException:
            os.close(child)
            raise
    return child


def _open_component(p, cur, create, from_link):
    """Open p as O_DIRECTORY from cur.  Create if create and not from_link.

    Returns fd on success, None on ENOTDIR/ELOOP (symlink indicator).
    Raises on FileNotFound when cannot create, or other errors.
    """
    try:
        return os.open(p, os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur)
    except FileNotFoundError:
        if not create or from_link:
            raise
        return _create_dir(p, cur)
    except OSError as e:
        if e.errno not in (errno.ENOTDIR, errno.ELOOP):
            raise
        return None


def _read_link_target(p, cur, link_count):
    """Read and validate symlink target.  Rejects  ..  in target.

    Returns (target, new_link_count).  Raises ProvisioningError on
    absolute target, NUL byte,  ..  component, or count > 40.
    """
    target = os.readlink(p, dir_fd=cur)
    if target.startswith("/") or "\0" in target:
        raise ProvisioningError("resolve_dir", None, "invalid symlink: %s -> %s" % (p, target), None)
    for tp in target.split("/"):
        if tp == "..":
            raise ProvisioningError("resolve_dir", None, ".. in symlink target: %s -> %s" % (p, target), None)
    lc = link_count + 1
    if lc > _MAX_SYMLINK_COUNT:
        raise ProvisioningError("resolve_dir", None, "too many symlinks (>%d)" % _MAX_SYMLINK_COUNT, None)
    return target, lc


def _resolve_dir_fd(anchor_fd, rel, create=False):
    """Traverse rel from anchor_fd, resolving symlinks manually.

    Uses a tagged queue of (component, from_symlink) tuples.  Initial
    components are tagged False.  Symlink target components are tagged
    True and never auto-created.  ..  is rejected in all cases.
    """
    parts = []
    for p in rel.split("/"):
        if p == "..":
            raise ProvisioningError("resolve_dir", None, ".. in path: %s" % rel, None)
        if p and p != ".":
            parts.append((p, False))
    if not parts:
        return os.dup(anchor_fd)

    cur = anchor_fd
    link_count = 0
    i = 0
    ok = False
    try:
        while i < len(parts):
            p, from_link = parts[i]
            i += 1
            child = _open_component(p, cur, create, from_link)
            if child is None:
                target, link_count = _read_link_target(p, cur, link_count)
                new_parts = [(tp, True) for tp in target.split("/") if tp and tp != "."]
                parts = new_parts + parts[i:]
                i = 0
                continue
            if cur != anchor_fd:
                os.close(cur)
            cur = child
        ok = True
        return cur
    finally:
        if not ok and cur != anchor_fd:
            try:
                os.close(cur)
            except OSError:
                pass


# ========== secure temp file in a dir_fd ==========


def _safe_temp_at(dir_fd, prefix, temp_dir=None):
    """Create O_CREAT|O_EXCL|O_NOFOLLOW temp file in dir_fd.

    Returns (fd, rel_name).  Raises ProvisioningError with temp_dir.
    """
    for _ in range(10):
        name = "%s_%s_%d.tmp" % (prefix, _rand_hex(), os.getpid())
        try:
            fd = os.open(
                name,
                os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                mode=stat.S_IRUSR | stat.S_IWUSR,
                dir_fd=dir_fd,
            )
            return fd, name
        except FileExistsError:
            continue
    raise ProvisioningError("safe_temp", None, "cannot create temp file", temp_dir)


def _atomic_write(dst_dir_fd, rel, data, mode, temp_dir=None):
    """Write data (bytes) atomically into dir_fd, then fsync dir.

    Steps: temp -> write_all -> fchmod -> fsync -> replace -> fsync dir.
    Cleans up temp file on error.  temp_dir passed to errors.
    """
    fd, tmp = _safe_temp_at(dst_dir_fd, ".tmp_atomic_", temp_dir)
    try:
        _write_all(fd, data)
        os.fchmod(fd, mode)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp, rel, src_dir_fd=dst_dir_fd, dst_dir_fd=dst_dir_fd)
        _fsync_dir_fd(dst_dir_fd)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp, dir_fd=dst_dir_fd)
        except OSError:
            pass
        raise


def _fsync_dir_fd(fd):
    """Fsync an open directory fd.  Propagates OSError."""
    os.fsync(fd)


# ========== wheel-only gate ==========


def _is_wheel_install(pkg_dir):
    return os.path.isfile(os.path.join(pkg_dir, _WHEEL_MARKER))


# ========== ASCEND_HOME_PATH ==========


def _check_ascend_home():
    raw = os.environ.get("ASCEND_HOME_PATH")
    if raw is None:
        return None, True
    stripped = raw.strip()
    if not stripped:
        raise ProvisioningError("check_ascend_home", None, "ASCEND_HOME_PATH is empty or blank", None)
    if not os.path.isdir(stripped):
        raise ProvisioningError(
            "check_ascend_home", None, "ASCEND_HOME_PATH=%s is not a valid directory" % stripped, None
        )
    return stripped, False


# ========== ACL version gate ==========


def _check_acl_version():
    """Check ACL version >= 1.17. Return True/False; never raise."""
    try:
        import acl
    except Exception:
        return False
    try:
        result = acl.get_version()
    except Exception:
        return False
    try:
        major, minor, _patch, ret = result
    except (TypeError, ValueError):
        return False
    try:
        return ret == 0 and major >= 1 and minor >= 17
    except TypeError:
        return False


# ========== SoC detection ==========


def _detect_soc():
    try:
        import acl
    except Exception:
        return None
    try:
        soc = acl.get_soc_name()
    except Exception:
        return None
    if not isinstance(soc, str) or not soc:
        return None
    return soc


def _is_ascend950_soc(soc_name):
    if not isinstance(soc_name, str):
        return False
    return "Ascend910_95" in soc_name or "Ascend950" in soc_name


# ========== git identity ==========


def _parse_git_text(text):
    """Extract exactly one non-empty git: value from text; else None.

    Prefix must be 'git:' then only horizontal whitespace [ \\t]*,
    then a single non-empty value without whitespace on the same line.
    Cross-line values, leading/trailing whitespace in value rejected.
    """
    m = re.findall(r"^git:[ \t]*(\S+)$", text, re.MULTILINE)
    if len(m) != 1:
        return None
    return m[0]


def _read_git_field(path):
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return _parse_git_text(text)


def _read_git_field_fd(fd):
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None
        raw = os.read(fd, max(st.st_size, 65536))
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return _parse_git_text(text)


# ========== version comparison ==========


def _needs_provisioning(pkg_dir, config_fd):
    """Compare git fields.  Wheel invalid -> fail.  Target invalid -> needs."""
    wheel = _read_git_field(os.path.join(pkg_dir, "VERSION"))
    if wheel is None:
        raise ProvisioningError("check_version", None, "wheel VERSION git field missing or malformed", None)
    try:
        vfd = os.open(_VERSION_FILENAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=config_fd)
    except FileNotFoundError:
        return True
    try:
        target = _read_git_field_fd(vfd)
    finally:
        os.close(vfd)
    return target is None or wheel != target


# ========== lock (fcntl flock, config dir_fd anchored) ==========


def _acquire_lock(config_fd, deadline):
    """Acquire exclusive flock on the permanent lock file in config_fd."""
    while True:
        try:
            fd = os.open(
                _LOCK_FILENAME,
                os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                mode=stat.S_IRUSR | stat.S_IWUSR,
                dir_fd=config_fd,
            )
        except OSError as e:
            raise ProvisioningError("acquire_lock", None, "open fail (errno=%s): %s" % (e.errno, e), None)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                _close_fds_safe(fd)
                raise ProvisioningError("acquire_lock", None, "lock file is not regular", None)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            _close_fds_safe(fd)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProvisioningError("acquire_lock", None, "timeout after %ds" % _LOCK_TIMEOUT_SEC, None)
            time.sleep(min(remaining, 1.0))
        except OSError as e:
            _close_fds_safe(fd)
            raise ProvisioningError("acquire_lock", None, "flock fail (errno=%s): %s" % (e.errno, e), None)


def _release_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


# ========== INI (block-boundary, dir_fd) ==========


def _strip_line_ending(line):
    """Remove trailing \\r?\\n once, or just trailing \\r."""
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith("\n"):
        return line[:-1]
    return line.rstrip("\r")


def _filter_ini_lines(lines):
    """Remove all blocks whose header matches 'name:cann-hybm-compat.tar.gz'.

    Each line is stripped of its line ending before comparison.
    Header matching uses lstrip() to ignore leading whitespace.
    Block ends at next line whose stripped form starts with 'name:' or EOF.
    Appends the standard block at end.  Output is byte-stable (idempotent).
    """
    stripped = [_strip_line_ending(ln) for ln in lines]
    out = []
    i = 0
    while i < len(stripped):
        if stripped[i].lstrip() == "name:cann-hybm-compat.tar.gz":
            i += 1
            while i < len(stripped) and not stripped[i].lstrip().startswith("name:"):
                i += 1
        else:
            out.append(stripped[i])
            i += 1
    out.append(_INI_STANDARD_BLOCK)
    return out


def _update_ini_at(conf_dir_fd, temp_dir=None):
    """Atomically update INI via dir_fd. temp_dir used for errors."""
    old_stripped = []
    orig_st = None
    try:
        ifd = os.open(_INI_FILENAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=conf_dir_fd)
    except FileNotFoundError:
        old_stripped = []
    else:
        try:
            st = os.fstat(ifd)
            if stat.S_ISREG(st.st_mode):
                raw = os.read(ifd, max(st.st_size, 1048576))
                old_stripped = raw.decode("utf-8").splitlines(keepends=False)
                orig_st = st
        finally:
            os.close(ifd)
    new_lines = _filter_ini_lines(old_stripped)
    data = "\n".join(new_lines).encode("utf-8")
    fd, tmp = _safe_temp_at(conf_dir_fd, ".tmp_ini_", temp_dir)
    try:
        _write_all(fd, data)
        if orig_st is not None:
            os.fchown(fd, orig_st.st_uid, orig_st.st_gid)
            os.fchmod(fd, stat.S_IMODE(orig_st.st_mode))
        else:
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp, _INI_FILENAME, src_dir_fd=conf_dir_fd, dst_dir_fd=conf_dir_fd)
        _fsync_dir_fd(conf_dir_fd)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp, dir_fd=conf_dir_fd)
        except OSError:
            pass
        raise


# ========== tar verification ==========


def _verify_tar(path):
    """Strict tar verification: exactly two members.

    - directory entry 'aicpu_kernels_device' (no trailing slash in Python tarfile)
    - regular file 'aicpu_kernels_device/libcann_hybm_kernel.so'
    Rejects empty tar, duplicates, FIFO/device/links/absolute/traversal.
    """
    try:
        with tarfile.open(path, "r:gz") as tf:
            members = tf.getmembers()
    except (tarfile.TarError, OSError):
        return False
    if len(members) != 2:
        return False
    # Check the set of names: exactly the two expected.
    names = set(m.name for m in members)
    if names != {_TAR_EXPECTED_DIR, _TAR_EXPECTED_FILE}:
        return False
    for m in members:
        if m.name == _TAR_EXPECTED_DIR and not m.isdir():
            return False
        if m.name == _TAR_EXPECTED_FILE and not m.isfile():
            return False
        if m.issym() or m.islnk() or m.ischr() or m.isblk() or m.isfifo():
            return False
        if m.name.startswith("/") or ".." in m.name.split("/"):
            return False
    return True


# ========== JSON verification ==========


def _verify_json(path):
    try:
        with open(path, "r") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False


# ========== build kernel (CMake) ==========


def _build_kernel(pkg_dir, build_dir, ascend_home, temp_dir, deadline):
    """CMake configure + build, sharing one deadline."""
    ops = os.path.join(pkg_dir, "_ops")
    hsrc = os.path.join(pkg_dir, "_hybm_src")
    common = [
        "-DHYBM_KERNEL_PROJECT_ROOT=" + pkg_dir,
        "-DPROJECT_HYBM_SRC_BASE=" + hsrc,
        "-DASCEND_HOME_PATH=" + ascend_home,
        "-DCMAKE_BUILD_TYPE=RELEASE",
    ]

    def _run(args, stage):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProvisioningError(stage, None, "timed out (deadline exceeded)", temp_dir)
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "")[:_STDERR_TRUNC]
            err = (exc.stderr or "")[:_STDERR_TRUNC]
            detail = "stdout: %s, stderr: %s" % (out, err) if out else err
            raise ProvisioningError(stage, None, "timed out: %s" % detail, temp_dir)
        if proc.returncode != 0:
            raise ProvisioningError(stage, proc.returncode, proc.stderr[:_STDERR_TRUNC], temp_dir)

    _run(["cmake", "-S", ops, "-B", build_dir] + common, "cmake_configure")
    _run(["cmake", "--build", build_dir, "--target", "cann_hybm_kernel"], "cmake_build")


# ========== install artifacts (all dir_fd anchored) ==========


def _install_content_at(dir_fd, rel, content, mode, temp_dir=None):
    """Write content (str) atomically to dir_fd/rel with mode."""
    _atomic_write(dir_fd, rel, content.encode("utf-8"), mode, temp_dir)


def _copy_file_via_fd(dir_fd, rel, src_path, mode, temp_dir=None):
    """Copy src_path into dir_fd/rel atomically with mode via fd."""
    with open(src_path, "rb") as sf:
        data = sf.read()
    _atomic_write(dir_fd, rel, data, mode, temp_dir)


def _install_artifacts(pkg_dir, build_dir, kernel_fd, config_fd, conf_fd, temp_dir):
    """Install tar, JSON, INI, version via pre-opened dir fds.

    Raises ProvisioningError with temp_dir on any failure.
    """
    tar_src = os.path.join(build_dir, "hybm_kernel", _TAR_FILENAME)
    json_src = os.path.join(pkg_dir, "_ops", "hybm_kernel", _JSON_FILENAME)
    if not os.path.isfile(tar_src):
        raise ProvisioningError("install", None, "tar not found: %s" % tar_src, temp_dir)
    if not _verify_tar(tar_src):
        raise ProvisioningError("install", None, "tar verification failed", temp_dir)
    if not os.path.isfile(json_src):
        raise ProvisioningError("install", None, "json not found: %s" % json_src, temp_dir)
    if not _verify_json(json_src):
        raise ProvisioningError("install", None, "json validation failed", temp_dir)

    wheel_git = _read_git_field(os.path.join(pkg_dir, "VERSION"))
    if wheel_git is None:
        raise ProvisioningError("install", None, "wheel git field missing", temp_dir)

    _copy_file_via_fd(kernel_fd, _TAR_FILENAME, tar_src, stat.S_IRUSR | stat.S_IRGRP, temp_dir)
    _copy_file_via_fd(config_fd, _JSON_FILENAME, json_src, stat.S_IRUSR | stat.S_IRGRP, temp_dir)
    conf_mode = stat.S_IMODE(os.fstat(conf_fd).st_mode)
    os.fchmod(conf_fd, conf_mode | stat.S_IWUSR)
    try:
        _update_ini_at(conf_fd, temp_dir)
    finally:
        os.fchmod(conf_fd, conf_mode)
    ver = "mf version info:\nmf version: \ngit: %s\n" % wheel_git
    _install_content_at(config_fd, _VERSION_FILENAME, ver, stat.S_IRUSR | stat.S_IRGRP, temp_dir)


# ========== temp dir cleanup ==========


def _cleanup_temp_dir(temp_dir):
    try:
        shutil.rmtree(temp_dir)
    except OSError as e:
        raise ProvisioningError("cleanup", None, "install completed but cleanup failed: %s" % e, temp_dir)


# ========== orchestrator ==========


def _close_fds_safe(*fds):
    """Close one or more fds, ignoring OSError on each."""
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _provision_locked(pkg_dir, anchor_path, anchor, config_fd):
    """Run inside lock: recheck version, build, and install using fixed fds."""
    fds = []
    try:
        if not _needs_provisioning(pkg_dir, config_fd):
            return

        kernel_fd = _resolve_dir_fd(anchor, _KERNEL_REL, create=True)
        fds.append(kernel_fd)
        conf_fd = _resolve_dir_fd(anchor, _CONF_REL, create=True)
        fds.append(conf_fd)

        # Verify anchor inode hasn't been swapped after lock.
        cur_st = os.stat(anchor_path)
        anc_st = os.fstat(anchor)
        if (cur_st.st_dev, cur_st.st_ino) != (anc_st.st_dev, anc_st.st_ino):
            raise ProvisioningError("provision", None, "ASCEND_HOME_PATH changed after lock", None)

        deadline = time.monotonic() + _BUILD_TIMEOUT_SEC
        tmp = tempfile.mkdtemp(prefix="hybm_provision_")
        bdir = os.path.join(tmp, "build")
        try:
            _build_kernel(pkg_dir, bdir, anchor_path, tmp, deadline)
            _install_artifacts(pkg_dir, bdir, kernel_fd, config_fd, conf_fd, tmp)
            _cleanup_temp_dir(tmp)
        except ProvisioningError:
            raise
        except Exception as e:
            raise ProvisioningError("provision", None, str(e), tmp)
    finally:
        _close_fds_safe(*fds)


def provision():
    """Orchestrate wheel-only AICPU kernel provisioning."""
    pkg_dir = _get_pkg_dir()
    if not _is_wheel_install(pkg_dir):
        return

    ascend_home, skip = _check_ascend_home()
    if skip:
        return

    if not _check_acl_version():
        return

    soc = _detect_soc()
    if not _is_ascend950_soc(soc):
        return

    anchor_path, anchor = _open_anchor(ascend_home)
    config_fd = None
    try:
        try:
            config_fd = _resolve_dir_fd(anchor, _CONFIG_REL, create=True)
        except OSError as e:
            raise ProvisioningError(
                "resolve_config", None, "open/create config fail (errno=%s): %s" % (e.errno, e), None
            )
        needs = _needs_provisioning(pkg_dir, config_fd)
        if not needs:
            return

        lock_fd = _acquire_lock(config_fd, time.monotonic() + _LOCK_TIMEOUT_SEC)
        try:
            _provision_locked(pkg_dir, anchor_path, anchor, config_fd)
        finally:
            _release_lock(lock_fd)
    finally:
        _close_fds_safe(config_fd, anchor)
