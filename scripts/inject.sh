#!/usr/bin/env bash
# inject.sh <port> <seconds> orders|task <text>   -- after a delay, post an order or task to a running loop
sleep "$2"
if [ "$3" = orders ]; then
  curl -s -X POST -H 'content-type: application/json' -d "{\"orders\": [$(printf '%s' "$4" | .venv/bin/python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')]}" "http://127.0.0.1:$1/api/orders" >/dev/null
else
  curl -s -X POST -H 'content-type: application/json' -d "{\"text\": $(printf '%s' "$4" | .venv/bin/python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}" "http://127.0.0.1:$1/api/task" >/dev/null
fi
