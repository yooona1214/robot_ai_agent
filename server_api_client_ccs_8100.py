import requests
import json

def test_action_request():


    # FastAPI 서버의 엔드포인트 URL 
    # 8000번 포트
    
    
    url = "http://cd72-175-209-74-146.ngrok-free.app/current_service_start/robot_1/"  # 로컬에서 FastAPI 서버가 실행 중이어야 합니다.
    url = "http://cd72-175-209-74-146.ngrok-free.app/action_request2/robot_1/"  # 로컬에서 FastAPI 서버가 실행 중이어야 합니다.

    while True:
        # user_input = input("대화 입력: ")
        
        # # 테스트할 로봇 요청 데이터를 생성합니다.
        # test_request = {
        #     'poi_arg_list': [['poi1', 1,0,1], ['poi2', 1,0,2], ['poi3', 1,0,3] ]
        # }
        
        user_input = input("대화 입력: ")
        
        # 테스트할 로봇 요청 데이터를 생성합니다.
        test_request = {
            "robot_id": "robot_1",
            "user_query": user_input,
            "time_stamp": "2024-09-04T10:00:00Z",
            "loc_x": 1.0,
            "loc_y": 2.0
        }
        
        # POST 요청을 통해 서버에 데이터를 전송합니다.
        response = requests.get(url, json=test_request)
      
        # 서버로부터 받은 응답을 출력합니다.
        if response.status_code == 200:
            print("Test successful. Server response:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(f"Test failed with status code: {response.status_code}")

if __name__ == "__main__":
    # 테스트 함수 실행
    test_action_request()