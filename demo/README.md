# デモ

percentile-ranked triage を画面で確かめるために作ったデモです。再利用を前提には作っていません。

次の環境で動かしました：M5 MacBook Air 24GB（MPS）、Python 3.12、decider-ai 1.5.0。Mac（MPS）を前提にしたコードなので、ほかの環境で動かす場合は注意してください。

```bash
pip install decider-ai==1.5.0 torch
python demo/demo_server.py
```

起動後、http://localhost:8765 を開きます。初回はモデル（decider-0.8b、約 1.5 GB）をダウンロードします。
