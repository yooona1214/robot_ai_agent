import numpy as np
import pandas as pd
import os
import redis
import atexit
from dotenv import load_dotenv

from modules.agents import *
from modules.db_manager import *

# Redis 클라이언트 생성
r = redis.Redis(host='localhost', port=6379, db=0)

# Goal.json 파일 경로
GOAL_JSON_PATH = 'data/goal.json'

# DB manager 생성
dbmanger = DBManager(r)


# 종료 시 Redis 캐시를 비우도록 atexit에 등록
atexit.register(dbmanger.clear_redis_cache)


# task manager 값 설정
previous_poi_list = ['여자화장실', '윤명로_미술작품']
robot_x = -50.0,
robot_y = -70.0
robot_id = "robot_yna"

goal_agent = GoalInferenceAgent( dbmanger, GOAL_JSON_PATH)
# 대화 main loop
START = False

while True:

    user_input = input("입력: ")
    if not START:
        # 현재 날짜와 시간을 세션 ID로 설정
        # session_id = dbmanger.get_session_id()
        session_id = goal_agent.check_new_service(robot_id)
        START = True
    
    # Graph DB 연결
    current_agent, respond_goal_chat, intent = goal_agent.route(user_input, robot_x, robot_y, session_id)
    print("\n\n\n")
    print("~~~~~~~~~~~~~~~~~~~~~~~~")
    print(current_agent)
    print(respond_goal_chat)
    print(intent)

    
    # ### Replanning 돌려볼때 주석 해제
    # replanning_agent =  ReplanningAgent.get_instance(robot_id, dbmanger, GOAL_JSON_PATH)
    # session_id = replanning_agent.check_new_service(robot_id)
    # current_agent, respond_goal_chat, intent = replanning_agent.route(user_input, previous_poi_list, robot_x, robot_y, session_id)

    # print("\n\n\n")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(current_agent)
    # print(respond_goal_chat)
    # print(intent)