from semantic_router.encoders import OpenAIEncoder
from semantic_router import Route
from semantic_router.layer import RouteLayer

# Router
general_chat = Route(
    name = "general_chat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "넌 누구니",
        "넌 무슨 로봇이야",
        "일반적인 대화",
        "빈칸"
    ]
)

robot_control = Route(
    name = "robot_control",
    utterances=[
        "안내로봇에 관련된 질문",
        "작품소개에 관한 질문",
        '로봇 제어에 관한 질문',
        '로봇 서비스 요청에 관련된 질문',
        "1층에 있는 작품들을 설명해줘",
        "이 건물에 있는 작품들을 소개해줘",
        
    ]
)

routes = [general_chat, robot_control]

class Router:
    def __init__(self, encoder):
        self.rl = RouteLayer(encoder = encoder, routes=routes)
        self.force_robot_control = False
    
    def route(self, user_input):
        if user_input == "!다시":
            self.force_robot_control = False
            
        if self.force_robot_control:
            return "robot_control"
        
        # 유저 입력을 인코딩하고 적절한 라우트를 선택
        route = self.rl(user_input).name
        
        # 만약 선택된 라우트가 robot_control이면, 상태를 업데이트
        if route == "robot_control":
            self.force_robot_control = True
            
        elif route == None:
            route = "general_chat"
        
        return route
