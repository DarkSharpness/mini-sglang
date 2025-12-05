# Request Lifecycle Walkthrough: From API to Backend and Back

This document outlines the journey of a request through the `mini-sglang` system, detailing the interaction between the API Server, Tokenizer, and Backend (Scheduler).

Our system uses **ZeroMQ (ZMQ)** for asynchronous communication between components. The key data structures for these messages are defined in `python/minisgl/message/`.

## 1. API Server: Receiving the Request
**File:** `python/minisgl/server/api_server.py`

*   **Entry Point:** A user sends a request (e.g., `GenerateRequest`) to the FastAPI server.
*   **Processing:** The `FrontendManager` assigns a unique ID (`uid`) to the request.
*   **Message Creation:** The server creates a `TokenizeMsg` (defined in `message/tokenizer.py`).
    *   **Content:** `uid`, `text` (prompt), `output_len`, and `sampling_params`.
*   **Action:** The message is pushed to the **Tokenizer** via the `send_tokenizer` queue (ZMQ Push).

## 2. Tokenizer: Text to IDs
**File:** `python/minisgl/tokenizer/server.py`

*   **Reception:** The `tokenize_worker` listens on `recv_listener` (ZMQ Pull) and receives the `TokenizeMsg`.
*   **Tokenization:** The `TokenizeManager` converts the input text string into a tensor of token IDs (`input_ids`).
*   **Message Creation:** The worker wraps the result into a `UserMsg` (defined in `message/backend.py`).
    *   **Content:** `uid`, `input_ids`, `output_len`, and `sampling_params`.
*   **Action:** The message is pushed to the **Backend** via the `send_backend` queue (ZMQ Push).

## 3. Backend (Scheduler): Model Execution
**Files:** `python/minisgl/scheduler/scheduler.py` & `python/minisgl/scheduler/io.py`

*   **Reception:** The `Scheduler` (inheriting from `SchedulerIOMixin`) receives the `UserMsg` via `receive_msg` (ZMQ Pull).
*   **Execution:**
    *   The request is added to the `Engine`.
    *   The model runs (prefill/decode) and generates the next token ID.
*   **Result Processing:** In `_process_batch_result`, the scheduler takes the generated `next_token_id`.
*   **Message Creation:** The scheduler creates a `DetokenizeMsg` (defined in `message/tokenizer.py`).
    *   **Content:** `uid`, `next_token` (integer ID), and `finished` (boolean flag indicating if EOS was hit or max length reached).
*   **Action:** The message is pushed to the **Detokenizer** via `send_result` (ZMQ Push).
    *   *Note: The Detokenizer is simply a `tokenize_worker` instance listening on a specific address.*

## 4. Tokenizer (Detokenizer): IDs to Text
**File:** `python/minisgl/tokenizer/server.py`

*   **Reception:** The `tokenize_worker` receives the `DetokenizeMsg` via `recv_listener`.
*   **Detokenization:** The `DetokenizeManager` converts the `next_token` ID back into a string (`incremental_output`).
*   **Message Creation:** The worker wraps the result into a `UserReply` (defined in `message/frontend.py`).
    *   **Content:** `uid`, `incremental_output` (the generated text chunk), and `finished`.
*   **Action:** The message is pushed back to the **API Server** via the `send_frontend` queue (ZMQ Push).

## 5. API Server: Response to User
**File:** `python/minisgl/server/api_server.py`

*   **Reception:** The `FrontendManager` receives the `UserReply` via the `recv_tokenizer` queue (ZMQ Pull).
*   **Action:** The server matches the `uid` to the active request and streams the `incremental_output` back to the user (or accumulates it for a non-streaming response).

---

## Summary of Message Structures

All message definitions are located in `python/minisgl/message/`.

| Stage | Message Type | Source | Destination | Key Fields |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `TokenizeMsg` | API Server | Tokenizer | `uid`, `text`, `sampling_params` |
| **2** | `UserMsg` | Tokenizer | Backend | `uid`, `input_ids` (Tensor) |
| **3** | `DetokenizeMsg` | Backend | Tokenizer | `uid`, `next_token` (Int) |
| **4** | `UserReply` | Tokenizer | API Server | `uid`, `incremental_output` (Str) |
