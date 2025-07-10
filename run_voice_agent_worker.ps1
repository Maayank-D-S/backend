# run_voice_agent_worker.ps1
# --------------------------------------------------------
# go to project root (folder where this script lives)
Set-Location -Path $PSScriptRoot

# 1) activate the 3.11 venv
& .\venv\Scripts\Activate.ps1

# 2) load .env into the current process
(Get-Content ".env") | ForEach-Object {
    if ($_ -match '^\s*(#|$)') { return }
    $k,$v = $_ -split '=',2
    Set-Item -Path "Env:$($k.Trim())" -Value $v.Trim()
}

# 3) run LiveKit agent worker via module entry-point
$PY = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

& $PY -m livekit.agents run `
      --entrypoint voice_agent.voice_bot:entrypoint `
      --identity salesbot `
      --watch-prefix user_
