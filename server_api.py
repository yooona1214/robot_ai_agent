"""FAST API LLM Agent 서버와 연동 모듈"""

import json
import logging
from datetime import datetime

import requests

from fastapi import Request, FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import os
import redis
import atexit
from dotenv import load_dotenv

from task_manager import *
from modules.agents import *
from modules.agents_studay import *
from modules.router import *
from modules.db_manager import *



# FastAPI 서버 연결
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (배포 환경에서는 특정 도메인만 허용하도록 설정하는 것이 좋음)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)
# Redis 클라이언트 생성
r = redis.Redis(host='localhost', port=6379, db=0)

# Goal.json 파일 경로
GOAL_JSON_PATH = 'data/goal.json'
 
# DB manager 생성
dbmanager = DBManager(r)

###서비스가 끝나면 비워야 하는데 이건 좀 생각을 해보쟈####
# 종료 시 Redis 캐시를 비우도록 atexit에 등록
atexit.register(dbmanager.clear_redis_cache)


agents = {}

def save_and_clear_cache(robot_id):
    #######골 추론 종료 및 장기메모리 db에 저장/단기메모리 초기화
    # db에 현재 캐쉬메모리 저장
    dbmanager.save_conversations_to_postgresql(robot_id)
    # 챗히스토리 초기화
    # 캐쉬 메모리 삭제
    if r.exists(robot_id):  # 해당 키가 존재하는지 확인
        r.delete(robot_id)  # 해당 로봇 ID의 키 삭제
        print(f"로봇 ID: {robot_id.encode('utf-8')}의 REDIS 챗히스토리를 모두 삭제했습니다.")
    else:
        print(f"로봇 ID: {robot_id.encode('utf-8')}에 해당하는 REDIS 챗 히스토리가 없습니다.")

def get_or_create_agent(robot_id):
    if robot_id not in agents:
        # 에이전트가 없으면 생성 후 딕셔너리에 저장
        agent = GoalInferenceAgent(dbmanager, GOAL_JSON_PATH)
        agents[robot_id] = agent
        logging.info(f"Created new agent for robot_id: {robot_id}")
    else:
        agent = agents[robot_id]
        logging.info(f"Using existing agent for robot_id: {robot_id}")
    return agent

@app.get("/")
async def root():
    """get user messages"""
    return {"message": "ROBOT SERVER IS OPENED"}

# 1. 골 추론 요청
@app.post("/action_request")
async def chat(request: Request):
    """post"""

    robotrequest = await request.json()
    response_chat, intent, final_poi_list = response_chat_goal(request=robotrequest)
    
    response_data =  {"llm_output": response_chat , "intent" : int(intent), "poi_list": final_poi_list}
    response = json.dumps(response_data, ensure_ascii=False)
    
    return response




def response_chat_goal(request):
    """콜백 기반 LLM 응답"""

    # 로봇 데이터 파싱
    robot_id = request["robot_id"]
    user_input = request["user_query"]
    time_stamp = request["time_stamp"]
    robot_x = request["loc_x"]
    robot_y = request["loc_y"]
    
    # 로봇 id 별 에이전트 인스턴스 생성
    goal_infer_agent = get_or_create_agent(robot_id)
    
    # 첫 발화기준 로봇 세션 id 생성
    session_id = goal_infer_agent.check_new_service(robot_id)
    
    if user_input == "주행해줘":
        
        agent_response = "테스트입니다"
        intent = 3
        final_poi_list = [["1층-융기원-20240905154025_로봇AX솔루션", "1", "4", "2"], ["1층-융기원-20240905154025_스마트단말SW연구", "1", "4", "2"]]        
        
        
        return agent_response, intent, final_poi_list
    elif user_input == "!다시":
        # robot_id 에 맞는 task_manager 인스턴스 로드
        goal_infer_agent = get_or_create_agent(robot_id)
        goal_infer_agent.restart_service()
        
        # 메모리 저장 및 삭제
        save_and_clear_cache(robot_id)
        print("초기화 요청 완료")
        
        return "인스턴스, 레디스 초기화", 0 , []
    else:
        # 골 추론 에이전트 
        print("INPUT: ", user_input)
        agent_id, agent_response, intent = goal_infer_agent.route(user_input,robot_x, robot_y, session_id)
        
        if intent != 3:
            final_poi_list = []

        # intent가 3이면 goal_json 만들기
        if intent == 3:
            
            # robot_id 별로 task manager 선언
            task_manager = TaskManager.get_instance(robot_id)
            
            # [['로봇AX솔루션팀_사무실', '1', '4', '2']], ['로봇AX솔루션팀_사무실']
            goal_json_poi_list, only_poi_list = goal_infer_agent.get_poi_list()
            final_poi_list = only_poi_list

            # goal_agent에서 생성한 poi 설정값 리스트으로 current_service_start 서비스에 보낼 goal.json 생성
            goal_json = task_manager.generate_goal_json(goal_json_poi_list)
            
            # task manager에서 사용할 상태 테이블 생성
            task_manager.initialize_poi_state_dict(goal_json)
            
            #######골 추론 종료 및 장기메모리 db에 저장/단기메모리 초기화
            # 메모리 저장 및 삭제
            save_and_clear_cache(robot_id)
        
        print("=============================================")
        print("=============================================")
        print("=============================================")
        print("agent_id: ", agent_id, "\nagent_response:", agent_response, "\nintent:", intent, "\nfinal_poi_list:", final_poi_list)
        print("=============================================")
        
        return agent_response, intent, final_poi_list




# 3. current_service_start 요청
@app.get("/current_service_start/{robot_id}")
async def chat(robot_id: str,request: Request):
    """get"""

    # robot_id 에 맞는 task_manager 인스턴스 로드
    task_manager =  TaskManager.get_instance(robot_id)
    
    # poi_state_dict를 보고 현재 not_done인 가장 빠른 poi의 정보를 불러오기
    current_poi = task_manager.find_current_poi()
    
    if current_poi == None:
        response_data = "Error: No POI"
        response = json.dumps(response_data, ensure_ascii=False)
        return response
    
    else:
        # current_service_start 생성
        current_service_start = task_manager.load_current_service_start(current_poi)
        
        return current_service_start




# task_finished 요청
@app.get("/task_finished/{robot_id}")
async def chat(robot_id: str,request: Request):
    """post"""
    
    # robot_id 에 맞는 task_manager 인스턴스 로드
    task_manager =  TaskManager.get_instance(robot_id)
    current_poi = task_manager.find_current_poi()
    
    updated_poi_dict = task_manager.update_poi_state_dict(current_poi, "done")
    
    
    return {"Task finished 잘 왔당 헤헤"}



# service_cancel 요청
@app.get("/service_cancel/{robot_id}")
async def chat(robot_id: str,request: Request):
    """post"""
    
    # robot_id 에 맞는 task_manager 인스턴스 로드
    goal_infer_agent = get_or_create_agent(robot_id)
    goal_infer_agent.restart_service()
    
    # 메모리 저장 및 삭제
    save_and_clear_cache(robot_id)
    print("초기화 요청 완료")
    
    return {"초기화 돼찌롱 메롱"}



# replanning 요청
@app.post("/action_request2")
async def chat(request: Request):
    """post"""

    robotrequest = await request.json()
    response_chat, intent, final_poi_list = response_replanning_agent(request=robotrequest)
    
    response_data =  {"llm_output": response_chat , "intent" : int(intent), "poi_list": final_poi_list}
    response = json.dumps(response_data, ensure_ascii=False)
    
    return response


def response_replanning_agent(request):
    """콜백 기반 LLM 응답"""

    # 로봇 데이터 파싱
    robot_id = request["robot_id"]
    user_input = request["user_query"]
    time_stamp = request["time_stamp"]
    robot_x = request["loc_x"]
    robot_y = request["loc_y"]

    # 리플래닝 에이전트 해야하는 일
    # 1. task manager에서 현재 not_done인 poi들 불러오기
    # 2. replanning 에이전트는 not_done인거랑 user input을 입력으로 새로운 poi list 만들기
    # 3. 이때, 3개 병렬 에이전트로 구성되어야함
    
    # # P1. robot_id 에 맞는 task_manager 인스턴스 로드
    # task_manager =  TaskManager.get_instance(robot_id)
    # previous_poi_list = task_manager.find_current_poi()
    
    # task manager 값 설정 (Test)
    previous_poi_list = {'남자화장실': 'done', '여자화장실': 'not_done', '윤명로_미술작품': 'not_done'} 
    robot_x = -50.0,
    robot_y = -70.0
    robot_id = "robot_yna"
        
    # P2. replanninag 에이전트 : previous + user input으로 새로운 poi list 만들기    
    # 첫 발화기준 로봇 세션 id 생성 
    replanning_agent =  ReplanningAgent.get_instance(robot_id, dbmanager, GOAL_JSON_PATH)
    session_id = replanning_agent.check_new_service(robot_id)
    replanning_agent.route(user_input, previous_poi_list, robot_x, robot_y, session_id)
    
    
############################################################################################################
# 스투데이용 엔드포인트
@app.post("/studay")
async def chat(request: Request, background_tasks: BackgroundTasks):
    """post"""

    robotrequest = await request.json()
    
    background_tasks.add_task(response_studay, request=robotrequest)

    return {"version": "2.0", "useCallback": True}

agent = LLMagent()
agent.load_all()

def response_studay(request):
    
    """콜백 기반 LLM 응답"""

    # 카톡 내용 파싱
    user_id = request["userRequest"]["user"]["id"]
    message = request["userRequest"]["utterance"]
    callback_url = request["userRequest"]["callbackUrl"]

    print("USER: ", user_id, "\nMESSAGE: ",message, "\n URL: ", callback_url)

    response = agent.identify_user(user_id, message)
    
    print("RESPONSE: ", response)
    res_kakao = {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": response}}]},
    }
    
    response = requests.post(url=callback_url, json=res_kakao, timeout=15)
    return response

