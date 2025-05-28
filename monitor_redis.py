import redis
import json
import time

def print_conversation(session_id, r, seen_turn_count):
    all_turns = r.lrange(session_id, 0, -1)
    new_turns = all_turns[seen_turn_count:]

    if new_turns:
        print(f"=== 로봇 ID: {session_id.decode('utf-8')} ===")
        for turn in new_turns:
            print(json.loads(turn))
        print("=======================================")
    
    return len(all_turns)

def monitor_conversations(r):
    seen_sessions = {}
    
    while True:
        try:
            session_ids = r.keys()
            
            for session_id in session_ids:
                session_id_str = session_id.decode('utf-8')
                seen_turn_count = seen_sessions.get(session_id_str, 0)
                seen_turn_count = print_conversation(session_id, r, seen_turn_count)
                seen_sessions[session_id_str] = seen_turn_count
            
            time.sleep(5)  # 5초마다 갱신
        except KeyboardInterrupt:
            print("모니터링을 종료합니다.")
            break

if __name__ == "__main__":
    # Redis 클라이언트 초기화
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    monitor_conversations(r)