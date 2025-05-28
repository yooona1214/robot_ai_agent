# Robot AI Agent


## 실행방법 ##
## FastAPI로 로컬 서버 생성
python -m uvicorn server_api:app --host 0.0.0.0 --port 8000 --reload --log-level debug
python -m uvicorn server_api:app --host 0.0.0.0 --port 8100 --reload --log-level debug
python -m uvicorn server_api:app --host 0.0.0.0 --port 8200 --reload --log-level debug


## 클라이언트 파일(로봇입력)
python server_api_client_test.py
python server_api_client_8100.py
python server_api_client_8200.py

# ######################







## ngrok 토큰 등록
ngrok config add-authtoken 2amxjrTAq8tsD02o2Ds4JqgCges_2hTjD1iZVStiwptCWpZfn

## Ngrok로 서버 접속 주소 생성 
ngrok http 8000
ngrok http --domain=quick-busy-hamster.ngrok-free.app 8000


8000 quick-busy-hamster.ngrok-free.app
8100 d45b-14-52-91-70.ngrok-free.app
ngrok start --all


메인 파일
- main.py
- 입력값 받은 후 라우터 전달
- 디비 매니저의 턴 메세지 저장을 통해 매 대화 턴을 캐쉬메모리에 저장

모니터링
- monitor_redis.py
- 캐쉬 메모리 모니터링하는 파일

라우터
- modules/router.py
- semantic router를 사용하여 입력발화의 context에 따른 라우팅

디비 매니저
- modules/db_manager.py
- 턴별 대화 redis 캐쉬메모리 저장 및 관리
- 세션 종료 후 캐쉬메모리 장기메모리에 저장 
- postgreSQL db 확인 후, 없으면 생성
- 있으면 해당 db에 대화 세션별로 redis 메모리 저장
- 세션 종료 후 대화 redis 메모리 삭제

멀티 에이전트 
- modules/agents.py
- 실제 실행할 프롬프트기반 챗지피티들
-
