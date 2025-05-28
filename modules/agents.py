import json
import redis
import os
import shutil
import importlib.resources as pkg_resources
import ast
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path


from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools.google_serper.tool import GoogleSerperRun
from langchain.agents import Tool, AgentExecutor
from langchain.chains import *
from langchain_core.output_parsers import StrOutputParser

from modules.create_react_agent_w_history import (
    create_openai_functions_agent_with_history,
    create_openai_functions_agent_with_history_query
)
from modules.prompts import *
from modules.db_manager import *
from modules.tools import *
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

from langchain_community.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from typing import Type
from langchain.tools import BaseTool


# Redis 클라이언트 생성
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# 환경변수설정
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Robot AI Agent"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = (
    ""  # yooona
)
os.environ["GPT_MODEL"] = "gpt-3.5-turbo"

# LLM model
llm_4 = ChatOpenAI(model="gpt-4-0613")
llm_4_t = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
llm_4_o_m = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
llm_4_o = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
llm_o_1 =  ChatOpenAI(model="o1-mini", temperature=0) #o1-mini

# llm_4_o_m = ChatOpenAI(model="gpt-4o-mini")
llm_3_5 = ChatOpenAI(model="gpt-3.5-turbo-0125")

llm_google = llm_4_t
llm_goal_builder = llm_4_o
llm_reply_question = llm_4_t
llm_summary = llm_4_t

# path
csv_path2 = pkg_resources.files("robot_info").joinpath("floor_description_240912.csv")

###### GRAPH TOOL 주석 처리
# # Graph
# graph = Neo4jGraph(
#                 url="bolt://54.235.226.49:7687",  
#                 username="neo4j",  
#                 password="spool-race-odds"
#             )


class GoalInferenceAgent:
    def __init__(self, db_manager, goal_json_path):
        self.db_manager = db_manager
        self.base_goal_json_path = goal_json_path
        
        # 기본 로봇 셋팅
        self.chat_history = []
        self.poi_list = []
        self.new_service = False
        self.robot_id = None
        self.session_id = None
        self.goal_generated = None
        self.current_agent = "intent_agent" # 맨처음엔 goal_chat한테 가게 설정해야함
        self.ro_x = None
        self.ro_y = None
        self.goal_json = None
        self.summary_flag = False
    
        
        # 체인버전
        self.goal_builder_chain = (
            goal_builder_prompt | llm_goal_builder | StrOutputParser()
        )

    ###### GRAPH TOOL 주석 처리
    #     # Graph 체인

    #     self.chain_test = GraphCypherQAChain.from_llm(
    #     llm_4_o,
    #     graph=graph,
    #     verbose=True,
    #     prompt = cypher_generation_prompt
    #    )

    #     graph_tool = Tool(
    #     name="Graph",
    #     func=self.execute_graph_query,
    #     description="""KT 연구소에 대한 공간 정보들을 Graph DB화 한 데이터 입니다. 
    #     사용자의 발화에 맞게 정보를 검색한 후 검색한 결과를 바탕으로 자연어기반 답변을 생성합니다.
    #     """
    #     )
        
        # Agent 1: 대화 및 csv를 통한 list 생성, tool 사용 에이전트 버전
        
        rag_robot_info = create_vector_store_as_retriever2(
            csv_path=csv_path2,
            str1="KT_floor_Information_fot_Docent_Robot",
            str2="This is a data containing space or artwork name, position and description for space and artworks.",
        )

        rag_robot_info_ = create_vector_store_as_retriever(
            csv_path=csv_path2,
            str1="KT_floor_Information_fot_Docent_Robot",
            str2="작품 상세정보가 있는 데이터입니다.",
        )
        # 기존 리트리버 도구들과 함께 tool 리스트에 GraphTool 추가
        #tool_robot_info = [rag_robot_info2, graph_tool]
        
        tool_robot_info = [rag_robot_info]
        
        ###### GRAPH TOOL 주석 처리
        # tool_robot_info2 = [rag_robot_info, graph_tool]

        # print(f"tool_robot_info2: {tool_robot_info2}")  # 두 개의 도구가 포함되어 있는지 확인
        
        tool_robot_info_ = [rag_robot_info_]
    
        # Agent 0: 일반과 작품설명 분류 에이전트 정의
        intent_agent = create_openai_functions_agent_with_history(
            llm_goal_builder, tool_robot_info, intent_prompt
        )
        self.intent_executor = AgentExecutor(
            agent=intent_agent, tools=tool_robot_info, verbose=True
        )
        
        ######## GRAPH TOOL 주석 처리 tools = tool_robot_info2 임 원래
        # Agent 1: 채팅 에이전트 정의
        goal_chat_agent = create_openai_functions_agent_with_history(
            llm_4_o, tool_robot_info, goal_chat_prompt
        )
        self.goal_chat_executor = AgentExecutor(
            agent=goal_chat_agent, tools=tool_robot_info, verbose=True
        )

        # Agent 2: POI 리스트 생성 에이전트 정의
        generate_poi_list_agent = create_openai_functions_agent_with_history_query(
            llm_4_o, tool_robot_info_, generate_poi_list_prompt
        )
        self.generate_poi_list_executor = AgentExecutor(
            agent=generate_poi_list_agent, tools=tool_robot_info_, verbose=True, max_iterations = 5
        )

        # Agent 3: Goal 완료 확인 에이전트 정의
        goal_done_check_agent = create_openai_functions_agent_with_history(
            llm_goal_builder, tool_robot_info, goal_done_check_prompt
        )
        self.goal_done_check_executor = AgentExecutor(
            agent=goal_done_check_agent, tools=tool_robot_info, verbose=True
        )
        
        # Agent 4: poi_list 확인 에이전트 정의
        goal_validation_agent = create_openai_functions_agent_with_history(
            llm_goal_builder, tool_robot_info, goal_validation_prompt
        )
        self.goal_validation_executor = AgentExecutor(
            agent=goal_validation_agent, tools=tool_robot_info, verbose=True
        )

        # Agent 5: 요약 에이전트 정의
        summary_agent = create_openai_functions_agent_with_history(
            llm_goal_builder, tool_robot_info, goal_summary_prompt
        )
        self.summary_executor = AgentExecutor(
            agent=summary_agent, tools=tool_robot_info, verbose=True
        )
             

        # self.reply_question_agent = create_openai_functions_agent_with_history(llm_reply_question, [], reply_question_prompt)
        # self.reply_question_executor = AgentExecutor(agent=self.reply_question_agent, tools=[], verbose=True)

        # self.summary_agent = create_openai_functions_agent_with_history(llm_summary, [], summary_prompt)
        # self.summary_executor = AgentExecutor(agent=self.summary_agent, tools=[], verbose=True)
        
    def execute_graph_query(self,query):
        try:
            # result = self.chain_test.invoke({"question": query})
            result = self.chain_test.invoke({"query": query})
            
            if not result or result == "I don't know the answer.":
                # 쿼리 결과가 없으면 실행 중단
                return "결과를 찾을 수 없습니다. 다른 질문을 해주세요."
            return result
        except Exception as e:
            # 에러가 발생할 경우
            return f"쿼리 실행 중 문제가 발생했습니다: {str(e)}"
        

    def _cache_turn(
        self, session_id, agent_id, user_input, goal_data, additional_question=None
    ):
        """캐시 메모리에 대화 저장"""
        turn = {
            "user_input": user_input,
            "goal_data": goal_data,
            "additional_question": additional_question,
            "timestamp": str(datetime.now()),
        }
        self.db_manager.redis_client.rpush(session_id, json.dumps(turn))
        
    def check_new_service(self, robot_id):
        "맨 처음 발화가 들어온 시점으로 세션 id 자체 생성"
        if not self.new_service:
            self.session_id = self.db_manager.get_session_id()
            self.new_service = True
            self.robot_id = robot_id
        
        return self.session_id
    
    def restart_service(self):
        """세션 초기화"""
        self.chat_history = []
        self.poi_list = []
        self.new_service = False
        self.robot_id = None
        self.session_id = None
        self.goal_generated = None
        self.current_agent = "intent_agent" 
        self.ro_x = None
        self.ro_y = None
        self.goal_json = None
        
        
    def intent_agent(self, user_input, session_id):
        """채팅 에이전트 - 사용자 입력 처리"""
        response = self.intent_executor.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })

        # 불필요한 ```json 제거 및 JSON 디코딩
        intent = response["output"]
        
        return intent
    
    def respond_goal_chat_agent(self, user_input, session_id):
        """채팅 에이전트 - 사용자 입력 처리"""
        response = self.goal_chat_executor.invoke({
            "input": user_input,
            "chat_history": self.chat_history,
            "schema" : graph.schema
        })

        # 불필요한 ```json 제거 및 JSON 디코딩
        respond_goal_chat = response["output"]
        print("###respond_goal_chat_agent: ", respond_goal_chat)

        return respond_goal_chat
    
    def respond_generate_poi_list_agent(self, robot_x, robot_y, chat_history):
        """POI 리스트 생성 에이전트 - POI 이름, BGM 타입, LED 색상 및 제어 정보 포함"""
        response = self.generate_poi_list_executor.invoke({
            "robot_x": robot_x,
            "robot_y": robot_y,
            "chat_history": chat_history
        })
        

        # 불필요한 ```json 제거 및 JSON 디코딩
        output_data_cleaned = response['output']
        print("###respond_generate_poi_list_agent: ", output_data_cleaned)
        
        poi_list = output_data_cleaned

        return poi_list

    def respond_goal_done_check_agent(self, chat_history):
        """목표 완료 확인 에이전트"""
        goal_done = False
        response = self.goal_done_check_executor.invoke({
            "chat_history":chat_history
        })
        output_data_string = response['output']
        output_data_string = output_data_string.replace("True", "true").replace("False", "false")
        output_data_list = json.loads(output_data_string)
        output_data_cleaned = output_data_list[1]
        

        try:
            print("###respond_goal_done_check_agent: ", output_data_cleaned)
            goal_done = output_data_cleaned

        except json.JSONDecodeError as e:
            print(f"JSON 디코딩 오류3: {e}")

        return goal_done
    
    
    def response_goal_validation_agent(self, poi_list, chat_history):
        """목표 완료 확인 에이전트"""
        response = self.goal_validation_executor.invoke({
            "poi_list": poi_list,
            "chat_history":chat_history
            
        })

        output_data_cleaned = response['output'].replace("```json", "").replace("```", "").strip()
        print("###response_goal_validation_agent: ", output_data_cleaned)
        poi_list = output_data_cleaned

        return poi_list

    def respond_summary_agent(self,user_input, poi_list, chat_history):
        """요약 에이전트 - 최종 요약 응답 생성"""
        goal_generated = False
        response = self.summary_executor.invoke({
            "input": user_input,
            "poi_list": poi_list,
            "chat_history": chat_history
        })
        print("###respond_summary_agent: ", response["output"])
        output_data_cleaned = ast.literal_eval(response["output"])


        try:
            respond_goal_chat = output_data_cleaned[0][1]
            goal_generated = output_data_cleaned[1][1]

        except json.JSONDecodeError as e:
            print(f"JSON 디코딩 오류4: {e}")
            respond_goal_chat = "Summary not available."

        return respond_goal_chat, goal_generated
        
    
    def respond_goal_verify_agent(self, user_input):    
        ## 프롬프트 수정해서 에이전트 만들어야 함##
        
        return 
    
    
    def get_poi_list(self):
        ### 여기서 바꿔주자
        csv_file_path = './robot_info/floor_description.csv'
        df = pd.read_csv(csv_file_path)  
           
        self.poi_list = ast.literal_eval(self.poi_list)       
        
        real_poi_list = []
        for sublist in list(self.poi_list):
            key_to_find = sublist[0]
            matching_row = df[df['Name'] == key_to_find]
            id_value = matching_row.iloc[0]['ID']
            real_poi_list.append(id_value)
            
        goal_json_poi_list = self.poi_list
        print(goal_json_poi_list, real_poi_list)
        return goal_json_poi_list, real_poi_list
        
    def route(self, user_input, robot_x, robot_y, session_id):
        print("****************************************************")
        print("****************************************************")
        

        # 챗 히스토리 로드
        self.chat_history = self.db_manager.get_conversation_history(self.robot_id, session_id)  

        # 로봇 x,y좌표 초기화
        self.ro_x = robot_x
        self.ro_y = robot_y
        
        # 라우팅값 초기화
        intent = 0
        goal_done = False  # 목표 완료 여부를 추적
        self.goal_generated_flag = False
        
        while self.current_agent != "summary_agent":
            # Agent1: 의도파악 에이전트 실행
            intent = self.intent_agent(user_input, session_id) #1:일반, 2:작품설명
            if intent == str(2):
                """작품설명이어서 로봇으로 바로 값 전송"""
                respond_goal_chat = "작품 설명 완료"
                
                time_stamp = str(datetime.now())
                self.db_manager.add_turn(self.robot_id, self.session_id, time_stamp, user_input, respond_goal_chat, self.current_agent)
                
                print(f"의도2(미들웨어) : ", intent)
                print(f"respond_goal_chat : ", respond_goal_chat)
                return self.current_agent, respond_goal_chat, intent
            else:
                """골 추론 에이전트 돌릴 경우"""
                self.current_agent = "goal_chat_agent"
                print(f"의도1(우리) : ", intent)
            

            # 1. 채팅 에이전트 실행
            if self.current_agent == "goal_chat_agent":
                respond_goal_chat = self.respond_goal_chat_agent(user_input, session_id)
                # 챗 히스토리 저장
                time_stamp = str(datetime.now())
                self.db_manager.add_turn(self.robot_id, self.session_id,time_stamp, user_input, respond_goal_chat, self.current_agent)
                
                print("111111111111111111111111111111111111111111111111111111111111111111111")
                # 채팅 응답을 반환하고 다음 에이전트로 넘어감
                self.current_agent = "goal_done_check_agent"
            
            # 2. POI 리스트 생성 에이전트 실행
            if self.current_agent == "goal_done_check_agent":
                
                # 채팅 에이전트가 남긴 챗 히스토리 다시 로드
                self.chat_history = self.db_manager.get_conversation_history(self.robot_id, session_id)  
                goal_done = self.respond_goal_done_check_agent(self.chat_history)
                print("22222222222222222222222222222222222222222222222222222222222222222222222")
                
                
                if goal_done:
                    # 목표가 완료되었으면 Summary 에이전트로 이동
                    print("GOAL DONE: TRUE")                  
                    self.current_agent = "generate_poi_list_agent"
               
                else:
                    # 목표가 완료되지 않았으면 해당 기록 저장하고 다시 채팅 에이전트로 돌아감
                    print("GOAL DONE: FALSE")
                    # 에이전트 응답 결과 저장
                    time_stamp = str(datetime.now())
                    self.db_manager.add_turn(self.robot_id, self.session_id,time_stamp, user_input, goal_done, self.current_agent)                    
                    # 채팅에이전트로 라우팅
                    self.current_agent = "goal_chat_agent"
                    intent = 1
                    return self.current_agent , respond_goal_chat, intent  # 서버로 채팅 응답 전송 후 루프 계속
            
            # 3. POI 리스트 생성 에이전트 실행
            if self.current_agent == "generate_poi_list_agent":
                
                # 채팅 에이전트가 남긴 챗 히스토리 다시 로드
                self.chat_history = self.db_manager.get_conversation_history(self.robot_id, session_id)  
                self.poi_list = self.respond_generate_poi_list_agent(self.ro_x, self.ro_y, self.chat_history)
                print("22222222222222222222222222222222222222222222222222222222222222222222222")
                # validation하고 서머리 에이전트로 넘어감
                self.poi_list = self.response_goal_validation_agent(self.poi_list, self.chat_history) 
                self.current_agent = "summary_agent"
                
        # 4. Summary 에이전트 실행 (목표 완료 후)
        if self.current_agent == "summary_agent":
            self.chat_history = self.db_manager.get_conversation_history(self.robot_id, session_id)

            # 1단계: Summary 에이전트 실행 후 요약 질문 반환
            respond_goal_chat, goal_generated = self.respond_summary_agent(user_input, self.poi_list, self.chat_history)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            
            if not self.summary_flag: # False 처음 서머리 에이전트가 탈 경우
                # 첫 번째 단계에서는 goal_generated를 아직 체크하지 않고 요약 질문을 사용자에게 보냄 (None)
                self.current_agent = "summary_agent"
                self.summary_flag = True
                print("@@@@@@@@@@@@써머리처음")
                
                #db저장
                time_stamp = str(datetime.now())
                self.db_manager.add_turn(self.robot_id, self.session_id,time_stamp, user_input, goal_done, self.current_agent)
                return self.current_agent, respond_goal_chat, intent

            else:
                if goal_generated == False:
                # goal_generated가 False면 다시 대화 에이전트로 돌아감
                    respond_goal_chat = "기존 계획을 초기화 하겠습니다. 안내받고 싶은신 장소를 다시 처음부터 말씀해주세요."
                    print("~~~~~~~~~~~~~~~써머리두번째 부정적 답변받은 상황")

                    self.current_agent = "goal_chat_agent"
                    self.summary_flag = False
                    #db저장
                    time_stamp = str(datetime.now())
                    self.db_manager.add_turn(self.robot_id, self.session_id,time_stamp, user_input, goal_done, self.current_agent)
                    self.restart_service()
                    return self.current_agent, respond_goal_chat, intent  # intent = 1: 다시 채팅으로 돌아감


                else: # True
                    # goal_generated가 True면 안내를 시작하는 응답 반환
                    respond_goal_chat = "안내를 시작하겠습니다."
                    print("===============써머리두번째 긍정적 답변받은 상황")
                    self.current_agent = "END"
                    self.summary_flag = False
                    intent = 3
                    #db저장
                    time_stamp = str(datetime.now())
                    self.db_manager.add_turn(self.robot_id, self.session_id,time_stamp, user_input, goal_done, self.current_agent)
                    return self.current_agent, respond_goal_chat, intent  # intent = 3: 안내 시작



class ReplanningAgent:
    _instances = {}
    
    def __init__(self, robot_id, db_manager, goal_json_path):
        """
        Replanning 에이전트 클래스 초기화
        """
        self.db_manager = db_manager
        self.base_goal_json_path = goal_json_path
        
        # 기본 로봇 셋팅
        self.chat_history = []
        self.poi_list = []
        self.new_service = False
        self.robot_id = robot_id
        self.session_id = None
        self.goal_generated = None
        self.current_agent = "intent_agent" # 맨처음엔 goal_chat한테 가게 설정해야함
        self.ro_x = None
        self.ro_y = None
        self.goal_json = None
        self.summary_flag = False
    
        
        # 체인버전
        self.goal_builder_chain = (
            goal_builder_prompt | llm_goal_builder | StrOutputParser()
        )

        # RAG
        rag_robot_info2 = create_vector_store_as_retriever(
            csv_path=csv_path2,
            str1="KT_floor_Information_fot_Docent_Robot",
            str2="This is a data containing space or artwork name, position and description for space and artworks.",
        )
        
        tool_robot_info2 = [rag_robot_info2]
        tool_robot_info = tool_robot_info2

        
        # Agent 0: 일반과 작품설명 분류 에이전트 정의
        intent_agent = create_openai_functions_agent_with_history(
            llm_goal_builder, tool_robot_info, intent_prompt
        )
        self.intent_executor = AgentExecutor(
            agent=intent_agent, tools=tool_robot_info, verbose=True
        )
        
        # Agent 1: 채팅 에이전트 정의
        replanning_chat_agent = create_openai_functions_agent_with_history(
            llm_goal_builder, tool_robot_info, replanning_chat_prompt
        )
        self.replanning_chat_executor = AgentExecutor(
            agent=replanning_chat_agent, tools=tool_robot_info, verbose=True
        )

        # Agent 2: POI 리스트 생성 에이전트 정의
        replanning_generate_poi_list_agent = create_openai_functions_agent_with_history_query(
            llm_goal_builder, tool_robot_info, replanning_generate_poi_list_prompt
        )
        self.replanning_generate_poi_list_executor = AgentExecutor(
            agent=replanning_generate_poi_list_agent, tools=tool_robot_info, verbose=True
        )

        # Agent 3: 목표 완료 확인 에이전트 정의
        replanning_goal_done_check_agent = create_openai_functions_agent_with_history(
            llm_goal_builder, tool_robot_info, replanning_goal_done_check_prompt
        )
        self.replanning_goal_done_check_executor = AgentExecutor(
            agent=replanning_goal_done_check_agent, tools=tool_robot_info, verbose=True
        )
        
        # Agent 3: 목표 완료 확인 에이전트 정의
        replanning_goal_validation_agent = create_openai_functions_agent_with_history(
            llm_goal_builder, tool_robot_info, replanning_goal_validation_prompt
        )
        self.replanning_goal_validation_executor = AgentExecutor(
            agent=replanning_goal_validation_agent, tools=tool_robot_info, verbose=True
        )

        # Agent 4: 요약 에이전트 정의
        replanning_summary_agent = create_openai_functions_agent_with_history(
            llm_goal_builder, tool_robot_info, goal_summary_prompt
        )
        self.replanning_summary_executor = AgentExecutor(
            agent=replanning_summary_agent, tools=tool_robot_info, verbose=True
        )
             

        # self.reply_question_agent = create_openai_functions_agent_with_history(llm_reply_question, [], reply_question_prompt)
        # self.reply_question_executor = AgentExecutor(agent=self.reply_question_agent, tools=[], verbose=True)

        # self.summary_agent = create_openai_functions_agent_with_history(llm_summary, [], summary_prompt)
        # self.summary_executor = AgentExecutor(agent=self.summary_agent, tools=[], verbose=True)
    
    @classmethod
    def get_instance(cls, robot_id, db_manager, goal_json_path):
        """
        주어진 robot_id에 대한 ReplanningAgent 인스턴스를 반환합니다.
        :param robot_id: 로봇의 ID
        :return: ReplanningAgent 인스턴스
        """
        if robot_id not in cls._instances:
            # 새로운 인스턴스 생성 후 딕셔너리에 저장
            cls._instances[robot_id] = ReplanningAgent(robot_id, db_manager, goal_json_path)
        return cls._instances[robot_id]


    def _cache_turn(
        self, session_id, agent_id, user_input, goal_data, additional_question=None
    ):
        """캐시 메모리에 대화 저장"""
        turn = {
            "user_input": user_input,
            "goal_data": goal_data,
            "additional_question": additional_question,
            "timestamp": str(datetime.now()),
        }
        self.db_manager.redis_client.rpush(session_id, json.dumps(turn))
        
    def check_new_service(self, robot_id):
        "맨 처음 발화가 들어온 시점으로 세션 id 자체 생성"
        if not self.new_service:
            self.session_id = self.db_manager.get_session_id()
            self.new_service = True
            self.robot_id = robot_id
        
        return self.session_id
    
    def restart_service(self):
        """세션 초기화"""
        self.chat_history = []
        self.poi_list = []
        self.new_service = False
        self.robot_id = None
        self.session_id = None
        self.goal_generated = None
        self.current_agent = "intent_agent" 
        self.ro_x = None
        self.ro_y = None
        self.goal_json = None
        
        
    def intent_replanning_agent(self, user_input, session_id):
        """채팅 에이전트 - 사용자 입력 처리"""
        response = self.intent_executor.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })

        # 불필요한 ```json 제거 및 JSON 디코딩
        intent = response["output"]
        
        return intent
    
    def respond_replanning_chat_agent(self, user_input, previous_poi_list):
        """채팅 에이전트 - 사용자 입력 처리"""
        response = self.replanning_chat_executor.invoke({
            "input": user_input,
            "previous_poi_list" : previous_poi_list, 
            "chat_history": self.chat_history
        })

        # 불필요한 ```json 제거 및 JSON 디코딩
        respond_goal_chat = response["output"]
        print("###respond_replanning_chat_agent: ", respond_goal_chat)

        return respond_goal_chat
    
    def respond_replanning_generate_poi_list_agent(self, previous_poi_list, robot_x, robot_y, chat_history):
        """POI 리스트 생성 에이전트 - POI 이름, BGM 타입, LED 색상 및 제어 정보 포함"""
        response = self.replanning_generate_poi_list_executor.invoke({
            "previous_poi_list": previous_poi_list,
            "robot_x": robot_x,
            "robot_y": robot_y,
            "chat_history": chat_history
        })
        

        # 불필요한 ```json 제거 및 JSON 디코딩
        output_data_cleaned = response['output']
        print("###respond_replanning_generate_poi_list_agent: ", output_data_cleaned)
        
        poi_list = output_data_cleaned

        return poi_list

    def respond_goal_done_check_agent(self, chat_history):
        """목표 완료 확인 에이전트"""
        goal_done = False
        response = self.replanning_goal_done_check_executor.invoke({
            "chat_history":chat_history
        })
        output_data_cleaned = response['output']
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^: ", output_data_cleaned)
        output_data_cleaned = response['output'].replace("```json", "").replace("```", "").strip()
        output_data_cleaned = json.loads(output_data_cleaned)
        output_data_cleaned = output_data_cleaned["goal_done"]
        

        try:
            print("###respond_goal_done_check_agent: ", output_data_cleaned)
            goal_done = output_data_cleaned

        except json.JSONDecodeError as e:
            print(f"JSON 디코딩 오류3: {e}")

        return goal_done
    
    
    def response_replanning_goal_validation_agent(self, poi_list, chat_history):
        """목표 완료 확인 에이전트"""
        response = self.replanning_goal_validation_executor.invoke({
            "poi_list": poi_list,
            "chat_history":chat_history
            
        })

        output_data_cleaned = response['output'].replace("```json", "").replace("```", "").strip()
        print("###response_replanning_goal_validation_agent: ", output_data_cleaned)
        poi_list = output_data_cleaned

        return poi_list

    def respond_replanning_summary_agent(self,user_input, poi_list, chat_history):
        """요약 에이전트 - 최종 요약 응답 생성"""
        goal_generated = False
        response = self.replanning_summary_executor.invoke({
            "input": user_input,
            "poi_list": poi_list,
            "chat_history": chat_history
        })
        print("###respond_summary_agent: ", response)
        output_data_cleaned = ast.literal_eval(response["output"])


        try:
            respond_goal_chat = output_data_cleaned[0][1]
            goal_generated = output_data_cleaned[1][1]

        except json.JSONDecodeError as e:
            print(f"JSON 디코딩 오류4: {e}")
            respond_goal_chat = "Summary not available."

        return respond_goal_chat, goal_generated
        
    
    def respond_goal_verify_agent(self, user_input):    
        ## 프롬프트 수정해서 에이전트 만들어야 함##
        
        return 
    
    
    def get_poi_list(self):
        ### 여기서 바꿔주자
        self.poi_list = ast.literal_eval(self.poi_list)
        goal_json_poi_list = self.poi_list
        only_poi_list = [sublist[0] for sublist in list(self.poi_list)]
        print(goal_json_poi_list, only_poi_list)
        return goal_json_poi_list, only_poi_list
        
    def route(self, user_input, previous_poi_list, robot_x, robot_y, session_id):
        print("****************REPLANNING STARTED*************************")
        print("***********************************************************")
        

        # 챗 히스토리 로드
        self.chat_history = self.db_manager.get_conversation_history(self.robot_id, session_id)  

        # 로봇 x,y좌표 초기화
        self.ro_x = robot_x
        self.ro_y = robot_y
        
        # 라우팅값 초기화
        intent = 0
        goal_done = False  # 목표 완료 여부를 추적
        self.goal_generated_flag = False
        
        while self.current_agent != "summary_agent":
            # Agent1: 의도파악 에이전트 실행
            # intent = self.intent_replanning_agent(user_input, session_id) #1:일반, 2:작품설명
            intent = 1
            
            if intent == str(2):
                """작품설명이어서 로봇으로 바로 값 전송"""
                respond_goal_chat = "작품 설명 완료"
                
                time_stamp = str(datetime.now())
                self.db_manager.add_turn(self.robot_id, self.session_id, time_stamp, user_input, respond_goal_chat, self.current_agent)
                
                print(f"의도2(미들웨어) : ", intent)
                print(f"respond_goal_chat : ", respond_goal_chat)
                return self.current_agent, respond_goal_chat, intent
            else:
                """골 추론 에이전트 돌릴 경우"""
                self.current_agent = "goal_chat_agent"
                print(f"의도1(우리) : ", intent)
            

            # 1. 채팅 에이전트 실행
            if self.current_agent == "goal_chat_agent":
                respond_goal_chat = self.respond_replanning_chat_agent(user_input, previous_poi_list)
                # 챗 히스토리 저장
                time_stamp = str(datetime.now())
                self.db_manager.add_turn(self.robot_id, self.session_id,time_stamp, user_input, respond_goal_chat, self.current_agent)
                
                print("111111111111111111111111111111111111111111111111111111111111111111111")
                # 채팅 응답을 반환하고 goal_done 에이전트로 넘어감
                self.current_agent = "goal_done_check_agent"
            

            # 2. 목표 완료 확인 에이전트 실행
            if self.current_agent == "goal_done_check_agent":
                # poi 리스트 생성 에이전트가 남긴 챗 히스토리 다시 로드
                self.chat_history = self.db_manager.get_conversation_history(self.robot_id, session_id)  
                goal_done = self.respond_goal_done_check_agent(self.chat_history)
 
                

                if goal_done:
                    # 목표가 완료되었으면 Summary 에이전트로 이동
                    print("GOAL DONE: TRUE")                  
                    self.current_agent = "generate_poi_list_agent"
                    
                else:
                    # 목표가 완료되지 않았으면 해당 기록 저장하고 다시 채팅 에이전트로 돌아감
                    print("GOAL DONE: FALSE")
                    # 에이전트 응답 결과 저장
                    time_stamp = str(datetime.now())
                    self.db_manager.add_turn(self.robot_id, self.session_id,time_stamp, user_input, goal_done, self.current_agent)                    
                    # 채팅에이전트로 라우팅
                    self.current_agent = "goal_chat_agent"
                    intent = 1
                    return self.current_agent , respond_goal_chat, intent  # 서버로 채팅 응답 전송 후 루프 계속
            
            # 3. POI 리스트 생성 에이전트 실행
            if self.current_agent == "generate_poi_list_agent":
                
                # 채팅 에이전트가 남긴 챗 히스토리 다시 로드
                self.chat_history = self.db_manager.get_conversation_history(self.robot_id, session_id)  
                self.poi_list = self.respond_replanning_generate_poi_list_agent(previous_poi_list, self.ro_x, self.ro_y, self.chat_history)
                print("22222222222222222222222222222222222222222222222222222222222222222222222")
                # validation하고 서머리 에이전트로 넘어감
                self.poi_list = self.response_replanning_goal_validation_agent(self.poi_list, self.chat_history) 
                self.current_agent = "summary_agent"
                
        # 4. Summary 에이전트 실행 (목표 완료 후)
        if self.current_agent == "summary_agent":
            self.chat_history = self.db_manager.get_conversation_history(self.robot_id, session_id)

            # 1단계: Summary 에이전트 실행 후 요약 질문 반환
            respond_goal_chat, goal_generated = self.respond_replanning_summary_agent(user_input, self.poi_list, self.chat_history)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            
            if not self.summary_flag: # False 처음 서머리 에이전트가 탈 경우
                # 첫 번째 단계에서는 goal_generated를 아직 체크하지 않고 요약 질문을 사용자에게 보냄 (None)
                self.current_agent = "summary_agent"
                self.summary_flag = True
                print("@@@@@@@@@@@@써머리처음")
                
                #db저장
                time_stamp = str(datetime.now())
                self.db_manager.add_turn(self.robot_id, self.session_id,time_stamp, user_input, goal_done, self.current_agent)
                return self.current_agent, respond_goal_chat, intent

            else:
                if goal_generated == False:
                # goal_generated가 False면 다시 대화 에이전트로 돌아감
                    respond_goal_chat = "기존 계획을 초기화 하겠습니다. 안내받고 싶은신 장소를 다시 처음부터 말씀해주세요."
                    print("~~~~~~~~~~~~~~~써머리두번째 부정적 답변받은 상황")

                    self.current_agent = "goal_chat_agent"
                    self.summary_flag = False
                    #db저장
                    time_stamp = str(datetime.now())
                    self.db_manager.add_turn(self.robot_id, self.session_id,time_stamp, user_input, goal_done, self.current_agent)
                    self.restart_service()
                    return self.current_agent, respond_goal_chat, intent  # intent = 1: 다시 채팅으로 돌아감


                else: # True
                    # goal_generated가 True면 안내를 시작하는 응답 반환
                    respond_goal_chat = "안내를 시작하겠습니다."
                    print("===============써머리두번째 긍정적 답변받은 상황")
                    self.current_agent = "END"
                    self.summary_flag = False
                    intent = 3
                    #db저장
                    time_stamp = str(datetime.now())
                    self.db_manager.add_turn(self.robot_id, self.session_id,time_stamp, user_input, goal_done, self.current_agent)
                    return self.current_agent, respond_goal_chat, intent  # intent = 3: 안내 시작
                