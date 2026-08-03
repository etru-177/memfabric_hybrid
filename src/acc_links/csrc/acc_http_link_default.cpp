/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "acc_http_link_default.h"

#include <cstring>
#include <ctime>

#include "acc_tcp_worker.h"

namespace ock {
namespace acc {

std::string AccHttpLinkDefault::ToLower(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

void AccHttpLinkDefault::ResetHttpState()
{
    httpState_ = AccHttpParseState::READ_REQUEST_LINE;
    recvBuf_.clear();
    method_.clear();
    uri_.clear();
    path_.clear();
    queryString_.clear();
    httpVersion_.clear();
    prevHeaderKey_.clear();
    prevHeaderVal_.clear();
    headers_.clear();
    contentLength_ = 0;
    needClose_ = false;
    lastActivity_ = std::time(nullptr);
}

bool AccHttpLinkDefault::IsIdleExpired()
{
    if (httpState_ == AccHttpParseState::COMPLETE || httpState_ == AccHttpParseState::READ_REQUEST_LINE) {
        if (std::time(nullptr) - lastActivity_ > static_cast<time_t>(HTTP_KEEPALIVE_TIMEOUT_S)) {
            LOG_INFO("Keep-alive idle timeout, closing " << ShortName());
            needClose_ = true;
        }
    }
    return needClose_;
}

Result AccHttpLinkDefault::Initialize(uint16_t sendQueueCap, int32_t workIndex, AccTcpWorker *worker)
{
    ASSERT_RETURN(sendQueueCap < UNO_256, ACC_INVALID_PARAM);
    ASSERT_RETURN(worker != nullptr, ACC_INVALID_PARAM);

    sendingQueue_ = AccMakeRef<AccLinkedMessageQueue>(sendQueueCap);
    ASSERT_RETURN(sendingQueue_.Get() != nullptr, ACC_NEW_OBJECT_FAIL);

    header_ = AccMsgHeader();
    workerIndex_ = static_cast<uint32_t>(workIndex);
    worker_ = worker;
    worker_->IncreaseRef();
    established_ = true;

    return ACC_OK;
}

void AccHttpLinkDefault::UnInitialize()
{
    sendingQueue_ = nullptr;
    if (worker_ != nullptr) {
        worker_->DecreaseRef();
        worker_ = nullptr;
    }
}

AccLinkedMessageNode *AccHttpLinkDefault::DequeueFront()
{
    ASSERT_RETURN(sendingQueue_.Get() != nullptr, nullptr);
    return sendingQueue_->DequeueFront();
}

Result AccHttpLinkDefault::EnqueueFront(AccLinkedMessageNode *node)
{
    ASSERT_RETURN(sendingQueue_.Get() != nullptr, ACC_NOT_INITIALIZED);
    return sendingQueue_->EnqueueFront(node);
}

AccLinkedMessageNode *AccHttpLinkDefault::TakeAwayMessages()
{
    ASSERT_RETURN(sendingQueue_.Get() != nullptr, nullptr);
    return sendingQueue_->TakeAwayMessages();
}

int AccHttpLinkDefault::FindLineEnd() const
{
    for (size_t i = 0; i + 1 < recvBuf_.size(); i++) {
        if (recvBuf_[i] == '\r' && recvBuf_[i + 1] == '\n') {
            return static_cast<int>(i);
        }
    }
    return -1;
}

/* ET mode: drain the socket buffer in one shot until EAGAIN or error.
 * Returns ACC_OK if any byte was read, ACC_LINK_EAGAIN if no data available,
 * ACC_LINK_ERROR on closed/broken connection. */
Result AccHttpLinkDefault::AppendRecvData()
{
    bool gotAny = false;
    char buf[HTTP_RECV_BUF_SIZE];
    while (true) {
        ssize_t n = PollInRecv(buf, sizeof(buf));
        if (n > 0) {
            recvBuf_.append(buf, static_cast<size_t>(n));
            gotAny = true;
            continue;
        }
        if (n == 0) {
            /* recv returns 0 when the peer closed the connection (FIN) */
            LOG_INFO("Connection closed by peer on " << ShortName());
            return ACC_LINK_ERROR;
        }
        auto errorNumber = errno;
        if (errorNumber == EAGAIN || errorNumber == EWOULDBLOCK) {
            break;
        }
        if (errorNumber == EINTR) {
            continue;
        }
        LOG_ERROR("Recv error on " << ShortName() << ", errno=" << errorNumber);
        return ACC_LINK_ERROR;
    }
    return gotAny ? ACC_OK : ACC_LINK_EAGAIN;
}

/* parse "METHOD /path HTTP/1.1\r\n" into method_, uri_, httpVersion_ */
Result AccHttpLinkDefault::ParseRequestLine()
{
    int lineEnd = FindLineEnd();
    if (lineEnd < 0) {
        if (recvBuf_.size() > HTTP_MAX_HEADER_SIZE) {
            LOG_ERROR("HTTP request line too long on " << ShortName() << ", bufSize=" << recvBuf_.size()
                                                       << ", max: " << HTTP_MAX_HEADER_SIZE);
            SendErrorResponse(AccHttpStatusCode::BAD_REQUEST);
            return ACC_LINK_MSG_INVALID;
        }
        return ACC_LINK_EAGAIN;
    }

    std::string line(recvBuf_, 0, static_cast<size_t>(lineEnd));
    recvBuf_.erase(0, static_cast<size_t>(lineEnd) + CRLF_LEN);

    RequestLineParser parsed(line);
    if (!parsed.valid) {
        LOG_ERROR("Invalid request line on " << ShortName() << ", line=" << line);
        SendErrorResponse(AccHttpStatusCode::BAD_REQUEST);
        return ACC_LINK_MSG_INVALID;
    }

    method_ = std::move(parsed.method);
    uri_ = std::move(parsed.uri);
    path_ = std::move(parsed.path);
    queryString_ = std::move(parsed.queryString);
    httpVersion_ = std::move(parsed.version);

    if (!IsKnownMethod(method_)) {
        LOG_ERROR("Unknown HTTP method on " << ShortName() << ", method=" << method_);
        SendErrorResponse(AccHttpStatusCode::BAD_REQUEST);
        return ACC_LINK_MSG_INVALID;
    }

    auto ret = ValidateHttpVersion();
    if (ret != ACC_OK) {
        return ret;
    }

    httpState_ = AccHttpParseState::READ_HEADERS;
    return ACC_OK;
}

/* validate httpVersion_ format and reject unsupported versions */
Result AccHttpLinkDefault::ValidateHttpVersion()
{
    if (httpVersion_.size() < HTTP_PREFIX_LEN || httpVersion_.compare(0, HTTP_PREFIX_LEN, "HTTP/") != 0) {
        LOG_ERROR("Invalid HTTP version on " << ShortName() << ", version=" << httpVersion_);
        SendErrorResponse(AccHttpStatusCode::BAD_REQUEST);
        return ACC_LINK_MSG_INVALID;
    }
    if (httpVersion_ != "HTTP/1.0" && httpVersion_ != "HTTP/1.1") {
        LOG_ERROR("Unsupported HTTP version on " << ShortName() << ", version=" << httpVersion_);
        SendErrorResponse(AccHttpStatusCode::BAD_REQUEST);
        return ACC_LINK_MSG_INVALID;
    }
    return ACC_OK;
}

/* parse header lines into headers_ map, stop at empty line, detect Content-Length */
Result AccHttpLinkDefault::ParseHeaders()
{
    while (true) {
        int lineEnd = FindLineEnd();
        if (lineEnd < 0) {
            if (recvBuf_.size() > HTTP_MAX_HEADER_SIZE) {
                LOG_ERROR("HTTP headers too long on " << ShortName() << ", bufSize=" << recvBuf_.size()
                                                      << ", max: " << HTTP_MAX_HEADER_SIZE);
                SendErrorResponse(AccHttpStatusCode::BAD_REQUEST);
                return ACC_LINK_MSG_INVALID;
            }
            return ACC_LINK_EAGAIN;
        }

        /* empty line marks end of headers */
        if (lineEnd == 0) {
            recvBuf_.erase(0, CRLF_LEN);
            return FinalizeHeaders();
        }

        std::string line(recvBuf_, 0, static_cast<size_t>(lineEnd));
        recvBuf_.erase(0, static_cast<size_t>(lineEnd) + CRLF_LEN);
        auto ret = ParseHeaderLine(line);
        if (ret != ACC_OK) {
            return ret;
        }
    }
}

/* finalize headers after the empty-line terminator */
Result AccHttpLinkDefault::FinalizeHeaders()
{
    /* flush any leftover folded value */
    if (!prevHeaderKey_.empty()) {
        headers_.emplace(prevHeaderKey_, prevHeaderVal_);
        prevHeaderKey_.clear();
        prevHeaderVal_.clear();
    }

    auto ret = ParseContentLength();
    if (ret != ACC_OK) {
        return ret;
    }

    /* check Connection: close from client */
    auto connIt = headers_.find("connection");
    if (connIt != headers_.end()) {
        std::string connVal = ToLower(connIt->second);
        if (connVal.find("close") != std::string::npos) {
            needClose_ = true;
        }
    }

    /* Transfer-Encoding: chunked is not supported; reject with 411 */
    auto teRange = headers_.equal_range("transfer-encoding");
    if (teRange.first != teRange.second) {
        LOG_ERROR("Transfer-Encoding not supported on " << ShortName() << ", method=" << method_);
        SendErrorResponse(AccHttpStatusCode::LENGTH_REQUIRED);
        return ACC_LINK_MSG_INVALID;
    }

    /* handle Expect: 100-continue after all header validations passed */
    auto expectIt = headers_.find("expect");
    if (expectIt != headers_.end() && expectIt->second == "100-continue") {
        char continueResp[] = "HTTP/1.1 100 Continue\r\n\r\n";
        PollOutWrite(continueResp, sizeof(continueResp) - 1);
    }

    if (contentLength_ > 0) {
        if (contentLength_ > maxBodySize_) {
            LOG_ERROR("HTTP body too large on " << ShortName() << ", contentLength=" << contentLength_
                                                << ", maxBodySize=" << maxBodySize_);
            SendErrorResponse(AccHttpStatusCode::PAYLOAD_TOO_LARGE);
            return ACC_LINK_MSG_INVALID;
        }
        httpState_ = AccHttpParseState::READ_BODY;
    } else {
        httpState_ = AccHttpParseState::COMPLETE;
    }
    return ACC_OK;
}

/* parse Content-Length header, reject duplicates with different values or overflow */
Result AccHttpLinkDefault::ParseContentLength()
{
    auto clRange = headers_.equal_range("content-length");
    if (clRange.first == clRange.second) {
        /* absent Content-Length means 0-length body per RFC 7230 §3.3.2 */
        return ACC_OK;
    }
    std::string firstCl = clRange.first->second;
    for (auto clIt = clRange.first; clIt != clRange.second; ++clIt) {
        if (clIt->second != firstCl) {
            LOG_ERROR("Multiple Content-Length headers with different values on " << ShortName());
            SendErrorResponse(AccHttpStatusCode::BAD_REQUEST);
            return ACC_LINK_MSG_INVALID;
        }
    }
    try {
        size_t pos;
        auto val = std::stoull(firstCl, &pos);
        if (pos != firstCl.size() || val > maxBodySize_) {
            LOG_ERROR("Invalid or too large Content-Length on " << ShortName() << ", contentLength=" << firstCl
                                                                << ", maxBodySize=" << maxBodySize_);
            SendErrorResponse(AccHttpStatusCode::PAYLOAD_TOO_LARGE);
            return ACC_LINK_MSG_INVALID;
        }
        contentLength_ = val;
    } catch (...) {
        LOG_ERROR("Invalid Content-Length on " << ShortName() << ", contentLength=" << firstCl);
        SendErrorResponse(AccHttpStatusCode::BAD_REQUEST);
        return ACC_LINK_MSG_INVALID;
    }
    return ACC_OK;
}

/* parse a single header line (folding, colon split, OWS trim, store) */
Result AccHttpLinkDefault::ParseHeaderLine(const std::string &line)
{
    /* obsolete line folding: line starting with space/tab continues previous header */
    if (!line.empty() && (line[0] == ' ' || line[0] == '\t')) {
        if (!prevHeaderKey_.empty()) {
            /* strip leading whitespace, append to previous value with single space */
            size_t firstNonSpace = line.find_first_not_of(" \t");
            if (firstNonSpace != std::string::npos) {
                prevHeaderVal_ += ' ' + line.substr(firstNonSpace);
            }
            return ACC_OK;
        }
        /* folding without a previous header is invalid */
        LOG_ERROR("Obsolete line folding without previous header on " << ShortName() << ", line=" << line);
        SendErrorResponse(AccHttpStatusCode::BAD_REQUEST);
        return ACC_LINK_MSG_INVALID;
    }

    /* flush previous folded header */
    if (!prevHeaderKey_.empty()) {
        headers_.emplace(prevHeaderKey_, prevHeaderVal_);
        prevHeaderKey_.clear();
        prevHeaderVal_.clear();
    }

    size_t colon = line.find(':');
    if (colon == std::string::npos) {
        LOG_ERROR("Header line missing colon on " << ShortName() << ", line=" << line);
        SendErrorResponse(AccHttpStatusCode::BAD_REQUEST);
        return ACC_LINK_MSG_INVALID;
    }

    std::string key = line.substr(0, colon);
    std::string val;
    if (colon + 1 < line.size()) {
        val = line.substr(colon + 1);
        /* trim leading OWS (spaces and tabs) */
        size_t first = val.find_first_not_of(" \t");
        if (first != std::string::npos) {
            val = val.substr(first);
        } else {
            val.clear();
        }
    }

    /* store with lowercase key */
    prevHeaderKey_ = ToLower(key);
    prevHeaderVal_ = val;
    return ACC_OK;
}

/* wait for recvBuf_ to accumulate contentLength_ bytes */
Result AccHttpLinkDefault::ParseBody()
{
    if (recvBuf_.size() >= contentLength_) {
        httpState_ = AccHttpParseState::COMPLETE;
        return ACC_OK;
    }
    return ACC_LINK_EAGAIN;
}

/*
 * Drive the HTTP state machine on each readable event.
 * Once COMPLETE, copy the body into data_ and return ACC_LINK_MSG_READY.
 */
Result AccHttpLinkDefault::HandlePollIn() noexcept
{
    if (needClose_) {
        LOG_ERROR("HandlePollIn skipped because needClose_ is set on " << ShortName());
        return ACC_LINK_ERROR;
    }
    auto ret = AppendRecvData();
    if (ret == ACC_LINK_ERROR) {
        return ret;
    }
    lastActivity_ = std::time(nullptr);

    if (httpState_ == AccHttpParseState::COMPLETE) {
        ResetForNextRequest();
    }

    if (ret == ACC_LINK_EAGAIN && recvBuf_.empty()) {
        return ACC_LINK_EAGAIN;
    }

    return ParseHttpMessage();
}

/* reset parse state for the next keep-alive request */
void AccHttpLinkDefault::ResetForNextRequest()
{
    httpState_ = AccHttpParseState::READ_REQUEST_LINE;
    method_.clear();
    uri_.clear();
    path_.clear();
    queryString_.clear();
    httpVersion_.clear();
    prevHeaderKey_.clear();
    prevHeaderVal_.clear();
    headers_.clear();
    contentLength_ = 0;
    needClose_ = false;
}

/* parse HTTP message through request-line/headers/body states */
Result AccHttpLinkDefault::ParseHttpMessage()
{
    while (true) {
        switch (httpState_) {
            case AccHttpParseState::READ_REQUEST_LINE: {
                auto ret = ParseRequestLine();
                if (ret != ACC_OK) {
                    return ret;
                }
                break;
            }
            case AccHttpParseState::READ_HEADERS: {
                auto ret = ParseHeaders();
                if (ret != ACC_OK) {
                    return ret;
                }
                break;
            }
            case AccHttpParseState::READ_BODY: {
                auto ret = ParseBody();
                if (ret != ACC_OK) {
                    return ret;
                }
                break;
            }
            case AccHttpParseState::COMPLETE: {
                auto bodyLen = static_cast<uint32_t>(contentLength_);
                auto bodyData = AccDataBuffer::Create(recvBuf_.data(), bodyLen);
                if (bodyData.Get() == nullptr) {
                    LOG_ERROR("Failed to allocate HTTP body buffer on " << ShortName() << ", bodyLen=" << bodyLen);
                    return ACC_MALLOC_FAIL;
                }
                recvBuf_.erase(0, contentLength_);
                data_ = bodyData;
                header_.bodyLen = bodyLen;
                header_.type = 0;
                httpState_ = AccHttpParseState::COMPLETE;
                return ACC_LINK_MSG_READY;
            }
        }
    }
}

/*
 * Dequeue the next message from the send queue and write it to the socket.
 * Nodes created by SendHttpResponse have headerRemain = 0 so the binary
 * AccMsgHeader is skipped — only HTTP response text is written on the wire.
 */
Result AccHttpLinkDefault::HandlePollOut(AccMsgHeader &header, AccDataBufferPtr &cbCtx) noexcept
{
    lastActivity_ = std::time(nullptr);
    ASSERT_RETURN(sendingQueue_.Get() != nullptr, ACC_NOT_INITIALIZED);
    AccLinkedMessageNode *oneMsg = sendingQueue_->DequeueFront();
    if (UNLIKELY(oneMsg == nullptr)) {
        return ACC_OK;
    }

    ASSERT_RETURN(!oneMsg->Sent(), ACC_OK);
    header = oneMsg->header;
    cbCtx = oneMsg->cbCtx;

    if (!oneMsg->HeaderSent()) {
        auto result = PollOutWrite(oneMsg->HeaderPtrToBeSend(), oneMsg->headerRemain);
        if (LIKELY(result > 0)) {
            if (!oneMsg->HeaderAllSent(result)) {
                sendingQueue_->EnqueueFront(oneMsg);
                return ACC_LINK_EAGAIN;
            }
        } else {
            delete oneMsg;
            oneMsg = nullptr;
            return SendPostProcess(errno);
        }
    }
    return WritePayload(oneMsg);
}

/* write the remaining payload (data) of a dequeued send node */
Result AccHttpLinkDefault::WritePayload(AccLinkedMessageNode *oneMsg)
{
    if (oneMsg->DataSent()) {
        delete oneMsg;
        oneMsg = nullptr;
        return ACC_LINK_MSG_SENT;
    }

    auto result = PollOutWrite(oneMsg->DataPtrToBeSend(), oneMsg->dataRemain);
    if (LIKELY(result > 0)) {
        if (!oneMsg->DataAllSent(result)) {
            sendingQueue_->EnqueueFront(oneMsg);
            return ACC_LINK_EAGAIN;
        }
        delete oneMsg;
        oneMsg = nullptr;
        return ACC_LINK_MSG_SENT;
    }
    delete oneMsg;
    oneMsg = nullptr;
    return SendPostProcess(errno);
}

bool AccHttpLinkDefault::HasBufferedRequest() const
{
    return httpState_ == AccHttpParseState::COMPLETE && !recvBuf_.empty();
}

bool AccHttpLinkDefault::IsKnownMethod(const std::string &method)
{
    static const char *known[] = {"GET", "POST", "PUT", "DELETE", "HEAD", "PATCH", "OPTIONS", "CONNECT", "TRACE"};
    for (auto m : known) {
        if (method == m) {
            return true;
        }
    }
    return false;
}

void AccHttpLinkDefault::SendErrorResponse(int16_t statusCode, const std::string &statusText)
{
    std::string resp = "HTTP/1.1 " + std::to_string(statusCode) + " " + statusText +
                       "\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    /* try to send error response directly; close regardless */
    PollOutWrite(resp.data(), static_cast<uint32_t>(resp.size()));
    needClose_ = true;
}

static std::string FormatHttpDate()
{
    std::time_t now = std::time(nullptr);
    struct tm tmBuf;
    gmtime_r(&now, &tmBuf);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%a, %d %b %Y %H:%M:%S GMT", &tmBuf);
    return buf;
}

/*
 * Build the full HTTP response (status line + headers + extra headers + blank line)
 * and enqueue it as AccLinkedMessageNodes with headerRemain = 0 so the binary
 * AccMsgHeader is not written to the wire.
 *
 * Enqueue order: header node first via EnqueueFront, then body node appended to
 * the tail via the queue's EnqueueBackNode, so the on-wire order is header then
 * body. If the body enqueue fails the header node stays (a header-only response
 * is sent and the connection is closed); if the header enqueue fails nothing is
 * queued, avoiding orphan body nodes.
 */
Result AccHttpLinkDefault::SendHttpResponse(int16_t statusCode, const std::string &statusText,
                                            const std::string &contentType, const AccDataBufferPtr &body,
                                            const std::map<std::string, std::string> &extraHeaders, bool noBody)
{
    lastActivity_ = std::time(nullptr);
    if (statusCode < HTTP_STATUS_CODE_MIN || statusCode > HTTP_STATUS_CODE_MAX) {
        LOG_ERROR("Invalid HTTP status code on " << ShortName() << ", statusCode=" << statusCode << ", min: "
                                                 << HTTP_STATUS_CODE_MIN << ", max: " << HTTP_STATUS_CODE_MAX);
        return ACC_INVALID_PARAM;
    }
    if (statusText.find('\r') != std::string::npos || statusText.find('\n') != std::string::npos) {
        LOG_ERROR("Status text contains CR/LF, rejecting on " << ShortName() << ", statusCode=" << statusCode);
        return ACC_INVALID_PARAM;
    }

    for (const auto &[key, value] : extraHeaders) {
        if (key.find('\r') != std::string::npos || key.find('\n') != std::string::npos ||
            value.find('\r') != std::string::npos || value.find('\n') != std::string::npos) {
            LOG_ERROR("Extra header key or value contains CR/LF, rejecting on " << ShortName()
                                                                                << ", statusCode=" << statusCode);
            return ACC_INVALID_PARAM;
        }
    }

    bool noBodyForStatus = (statusCode < HTTP_STATUS_CODE_1XX_END) || statusCode == HTTP_STATUS_NO_CONTENT ||
                           statusCode == HTTP_STATUS_NOT_MODIFIED;
    std::string headerStr =
        BuildResponseHeader(statusCode, statusText, contentType, body, extraHeaders, noBodyForStatus, noBody);
    if (headerStr.size() > static_cast<size_t>(UINT32_MAX)) {
        LOG_ERROR("Header string too large on " << ShortName() << ", headerSize=" << headerStr.size());
        return ACC_INVALID_PARAM;
    }
    auto headerBuf = AccDataBuffer::Create(headerStr.data(), static_cast<uint32_t>(headerStr.size()));
    if (headerBuf.Get() == nullptr) {
        LOG_ERROR("Failed to allocate header buffer in SendHttpResponse on " << ShortName()
                                                                             << ", headerSize=" << headerStr.size());
        return ACC_MALLOC_FAIL;
    }

    auto ret = EnqueueResponseNodes(headerBuf, body, noBodyForStatus, noBody, statusCode);
    if (ret != ACC_OK) {
        return ret;
    }

    return worker_->ModifyLink(this, POLLIN | POLLOUT | EPOLLET);
}

/* build the HTTP response header string (status line + headers + CRLF) */
std::string AccHttpLinkDefault::BuildResponseHeader(int16_t statusCode, const std::string &statusText,
                                                    const std::string &contentType, const AccDataBufferPtr &body,
                                                    const std::map<std::string, std::string> &extraHeaders,
                                                    bool noBodyForStatus, bool noBody)
{
    std::ostringstream oss;
    oss << "HTTP/1.1 " << statusCode << " " << statusText << "\r\n";

    if (!noBodyForStatus && !noBody) {
        oss << "Content-Type: " << contentType << "\r\n"
            << "Content-Length: " << (body.Get() ? body->DataLen() : 0) << "\r\n";
    } else {
        oss << "Content-Length: 0\r\n";
    }

    if (needClose_) {
        oss << "Connection: close\r\n";
    } else {
        bool isHttp10 = (httpVersion_ == "HTTP/1.0");
        auto connIt = headers_.find("connection");
        bool clientKeepAlive = false;
        if (connIt != headers_.end()) {
            clientKeepAlive = (ToLower(connIt->second).find("keep-alive") != std::string::npos);
        }
        if (isHttp10 && !clientKeepAlive) {
            oss << "Connection: close\r\n";
            needClose_ = true;
        } else {
            oss << "Connection: keep-alive\r\n"
                << "Keep-Alive: timeout=" << HTTP_KEEPALIVE_TIMEOUT_S << "\r\n";
        }
    }

    oss << "Date: " << FormatHttpDate() << "\r\n"
        << "Server: acc_links\r\n";

    for (const auto &[key, value] : extraHeaders) {
        oss << key << ": " << value << "\r\n";
    }
    oss << "\r\n";
    return oss.str();
}

/* enqueue header and body nodes into sendingQueue_
 * Pre-allocate all nodes before any enqueue, so that allocation failure
 * leaves the queue clean (no orphan node in the queue). */
Result AccHttpLinkDefault::EnqueueResponseNodes(const AccDataBufferPtr &headerBuf, const AccDataBufferPtr &body,
                                                bool noBodyForStatus, bool noBody, int16_t statusCode)
{
    ASSERT_RETURN(sendingQueue_.Get() != nullptr, ACC_NOT_INITIALIZED);
    ASSERT_RETURN(worker_ != nullptr, ACC_ERROR);

    bool needBody = !noBodyForStatus && !noBody && body.Get() != nullptr && body->DataLen() > 0;

    auto headerNode = new (std::nothrow) AccLinkedMessageNode;
    if (headerNode == nullptr) {
        LOG_ERROR("Failed to allocate header node on " << ShortName() << ", statusCode=" << statusCode);
        return ACC_NEW_OBJECT_FAIL;
    }
    headerNode->headerRemain = 0;
    headerNode->data = headerBuf;
    headerNode->dataRemain = headerBuf->DataLen();

    AccLinkedMessageNode *bodyNode = nullptr;
    if (needBody) {
        bodyNode = new (std::nothrow) AccLinkedMessageNode;
        if (bodyNode == nullptr) {
            LOG_ERROR("Failed to allocate body node on " << ShortName() << ", bodyLen=" << body->DataLen()
                                                         << ", statusCode=" << statusCode);
            delete headerNode;
            return ACC_NEW_OBJECT_FAIL;
        }
        bodyNode->headerRemain = 0;
        bodyNode->data = body;
        bodyNode->dataRemain = body->DataLen();
    }

    auto ret = sendingQueue_->EnqueueBackNode(headerNode);
    if (ret != ACC_OK) {
        LOG_ERROR("Failed to enqueue header node on " << ShortName() << ", result=" << ret
                                                      << ", statusCode=" << statusCode);
        delete headerNode;
        delete bodyNode;
        return ret;
    }

    if (bodyNode != nullptr) {
        ret = sendingQueue_->EnqueueBackNode(bodyNode);
        if (ret != ACC_OK) {
            LOG_ERROR("Failed to enqueue body node on "
                      << ShortName() << ", result=" << ret
                      << ", header already queued, will close, bodyLen=" << body->DataLen());
            delete bodyNode;
            needClose_ = true;
            (void)worker_->ModifyLink(this, POLLIN | POLLOUT | EPOLLET);
            return ret;
        }
    }
    return ACC_OK;
}

} // namespace acc
} // namespace ock
