---
name: merge-main
description: "Pull origin/main into the current branch and resolve conflicts. Preserve BaseACPClient; converge to main where safe."
allowed-tools:
  - read
  - edit
  - grep
  - exec
permissions:
  allow:
    - Exec(git)
    - Exec(python)
    - Read(**)
    - Write(**)
---

# 把 `origin/main` 合併到目前 branch

本 skill 適用於 `codex/devin-acp-provider` 這類以 `BaseACPClient` 重構為核心的 branch。合併時要不破壞 ACP 重構，同時在不影響功能的前提下盡量向 `main` 收斂。

## 環境約束

- 使用 PowerShell 執行 git/python 命令。
- 執行 Python 時優先使用 repo 的 `venv`：`\.venv\Scripts\python` 或 `\venv\Scripts\python`。
- 輸出使用繁體中文。
- commit message 最多 150 字元、用動詞開頭、說明原因與調整方式。

## 流程

### 1. 確認起點
- `git branch --show-current`
- `git status --short`
- 若 tree 已處於 half-merge 且無法繼續，才用 `git merge --abort`。

### 2. 拉取 main
- `git pull origin main`
- 若 fast-forward，用 `git log --oneline -5` 確認後結束。
- 若有衝突，繼續。

### 3. 盤點衝突
- `git diff --name-only --diff-filter=U` 列出所有衝突檔。
- 逐檔判斷是 ACP 專用檔還是通用檔：
  - ACP 專用：`agent/*_acp_client.py`、`agent/acp_client_base.py`、`agent/acp_client_factory.py`、`plugins/model-providers/*-acp/`、`tests/agent/test_*acp*.py`。
  - 通用：`hermes_cli/commands.py`、`gateway/run.py`、desktop/TS、docs。

### 4. 解決衝突

#### ACP 專用檔
- **絕對不能**把 `main` 的整份 `CopilotACPClient` 複製回來；必須保留 `BaseACPClient` subclass 架構。
- 把 `main` 的新行為整合成 subclass hook 或 module helper，例如：
  - `--acp` probe → `_pre_spawn_check`
  - gh-copilot deprecation → `_is_deprecation_message`
- `agent/acp_client_base.py` 只在「需要所有 ACP client 共用」時才加新 hook。
- 調整測試 patch 目標以符合重構後程式碼：
  - `subprocess.Popen` 從 `agent.copilot_acp_client` 改為 `agent.acp_client_base`
  - `_run_prompt` 改為 `_run_conversation_prompt`
- 與 copilot 無關的 ACP content-parts 測試搬到 `tests/agent/test_acp_client_base.py`。

#### 通用檔
- `hermes_cli/commands.py`：保留 Windows 安全的 `os.path.commonpath` skill-root 過濾；把 `main` 的 `get_project_skills_dirs()` 加入 `_allowed_roots`。
- `gateway/run.py`：同時保留 billing 與 connection error 兩組 regex；兩者是獨立功能。
- Desktop/TS：
  - 相鄰獨立測試用 `git merge-file -p --union` 合併，再補上缺少的 `})`。
  - 若 `main` 把單檔重構成目錄（如 `gateway-event.ts` → `gateway-event/`），接受 `main` 的目錄結構，`git rm` 舊單檔，只補遺失的關鍵行為。
- Docs：`website/docs/integrations/providers.md` union provider 表格與 ACP 段落。

### 5. 不破壞功能的前提下向 main 收斂
- 解決完衝突後，檢視本次異動：
  - 移除多餘的臨時檔（`.merged_*`、`.ours_*`、`.theirs_*`、`.union_*`、`.tmp_*.py`）。
  - 若測試檔因重構而新增非 copilot 專用測試，考慮搬到 `tests/agent/test_acp_client_base.py`。
  - 不要把核心 ACP 專用邏輯（如 `error_classifier.py` 的 ACP branch、`image_routing.py` 的 ACP 檢查）回退；它們是必要的功能。

### 6. 驗證
- 對每個修改過的 Python 檔跑 `python -m py_compile <file>`。
- 跑 `python -m pytest tests/agent/test_copilot_acp_client.py tests/agent/test_acp_client_base.py -q`。
- 若遇結構不相容的測試，修改測試（patch 對的方法）而不是回退重構。
- TypeScript 檔至少確認沒有 `<<<<<<<`/`=======`/`>>>>>>>` 殘留。

### 7. 提交
- 確認 `git diff --name-only --diff-filter=U` 無輸出。
- `git add` 所有解析檔與新增檔。
- Commit 範例：`Merge origin/main into <branch>. Resolve N conflicts: <策略摘要>。`
- 訊息必須 ≤150 字元、動詞開頭、說明原因與調整。

## 禁止
- 對 ACP 檔使用 `git checkout --theirs`。
- 在 Windows 上把 `hermes_cli/commands.py` 換成 `main` 的 `startswith` 前綴匹配，除非 skill 路徑已保證為 `/` 分隔。
- 移除 `gateway/run.py` 的 `_GATEWAY_BILLING_ERROR_RE`。
