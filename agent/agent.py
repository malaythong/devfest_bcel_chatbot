# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from langchain.globals import set_verbose
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import MemorySaver
from pytz import timezone

# ต้องแก้ import นี้ให้ตรงกับ project structure ของคุณ
# สมมติว่า react_graph.py และ tools.py อยู่ใน folder เดียวกัน
from .react_graph import create_graph
from .tools import initialize_tools

DEBUG = bool(os.getenv("DEBUG", default=False))
set_verbose(DEBUG)

# เปลี่ยนข้อความต้อนรับ
BASE_HISTORY = {
    "type": "ai",
    "data": {"content": "ສະບາຍດີ! BCEL ຍິນດີໃຫ້ບໍລິການ. ທ່ານຕ້ອງການສອບຖາມຂໍ້ມູນຜະລິດຕະພັນ ຫຼື ບໍລິການດ້ານໃດແດ່? (Hello! Welcome to BCEL. How can I assist you with our banking products and services today?)"},
}


class Agent:
    MODEL = "gemini-2.5-flash"

    _user_sessions: Dict[str, str]
    connector = None

    def __init__(self):
        self._user_sessions = {}
        self._langgraph_app = None
        self._checkpointer = None

    def user_session_exist(self, uuid: str) -> bool:
        return uuid in self._user_sessions

    # --- ส่วนที่ถูกตัดออก (Ticket Booking Logic) ---
    # เนื่องจากข้อมูลของคุณเป็น Products อย่างเดียว ไม่มีการจองตั๋ว
    # ผมจึง comment หรือตัด method พวก insert_ticket / decline_ticket ทิ้งไป
    # เพื่อลดความซับซ้อนและ error
    
    # async def user_session_insert_ticket(self, uuid: str) -> Any:
    #     return await self.user_session_invoke(uuid, None)

    # async def user_session_decline_ticket(self, uuid: str) -> dict[str, Any]:
    #     ...

    async def user_session_create(self, session: dict[str, Any]):
        """Create and load an agent executor with tools and LLM."""
        if self._langgraph_app is None:
            print("Initializing graph..")
            # แก้ไข initialize_tools ให้ไม่ต้อง return ticket functions
            # คุณต้องไปแก้ไฟล์ tools.py ด้วยนะครับ ให้ return แค่ tools list
            tools = await initialize_tools() 
            
            prompt = self.create_prompt_template()
            checkpointer = MemorySaver()
            
            # แก้ create_graph ให้รับ parameter น้อยลง (ตัด insert/validate ticket ออก)
            langgraph_app = await create_graph(
                tools,
                checkpointer,
                prompt,
                self.MODEL,
                DEBUG,
            )
            self._checkpointer = checkpointer
            self._langgraph_app = langgraph_app

        print("Initializing session")
        if "uuid" not in session:
            session["uuid"] = str(uuid.uuid4())
        session_id = session["uuid"]
        if "history" not in session:
            session["history"] = [BASE_HISTORY]
        history = self.parse_messages(session["history"])

        config = self.get_config(session_id)
        self._langgraph_app.update_state(config, {"messages": history})
        self._user_sessions[session_id] = ""

    async def user_session_invoke(
        self, uuid: str, user_prompt: Optional[str]
    ) -> dict[str, Any]:
        config = self.get_config(uuid)
        cur_message_index = (
            len(self._langgraph_app.get_state(config).values["messages"]) - 1
        )
        if user_prompt:
            user_query = [HumanMessage(content=user_prompt)]
            app_input = {"messages": user_query}
        else:
            app_input = None
        
        final_state = await self._langgraph_app.ainvoke(
            app_input,
            config=config,
        )
        messages = final_state["messages"]
        trace = self.retrieve_trace(messages[cur_message_index:])
        last_message = messages[-1]
        output = last_message.content
        
        response = {}
        response["output"] = output
        response["trace"] = trace
        
        # ตัด Logic การ confirm ticket ออก
        # if has_add_kwargs and last_message.additional_kwargs.get("confirmation"): ...
            
        response["state"] = final_state
        return response

    def retrieve_trace(self, messages: Sequence[BaseMessage]):
        trace = []
        for m in messages:
            if isinstance(m, ToolMessage):
                trace_info = {"tool_call_id": m.name, "results": m.content}
                add_kwargs = m.additional_kwargs
                if add_kwargs and add_kwargs.get("sql"):
                    trace_info["sql"] = add_kwargs.get("sql")
                trace.append(trace_info)
        return trace

    def user_session_reset(self, session: dict[str, Any], uuid: str):
        del session["history"]
        base_history = self.get_base_history(session)
        session["history"] = [base_history]
        history = self.parse_messages(session["history"])
        checkpoint = empty_checkpoint()
        config = self.get_config(uuid)
        self._checkpointer.put(
            config=config, checkpoint=checkpoint, metadata={}, new_versions={}
        )
        self._langgraph_app.update_state(config, {"messages": history})

    def set_user_session_header(self, uuid: str, user_id_token: str):
        self._user_sessions[uuid] = user_id_token

    def get_user_id_token(self, uuid: str) -> Optional[str]:
        return self._user_sessions.get(uuid)

    def create_prompt_template(self) -> ChatPromptTemplate:
        current_datetime = "Today's date and current time is {cur_datetime}."
        template = "\n\n".join(
            [
                PREFIX,  # ใช้ PREFIX ใหม่ด้านล่าง
                current_datetime,
                SUFFIX,
            ]
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", template), ("placeholder", "{messages}")]
        )
        prompt = prompt.partial(cur_datetime=self.get_datetime)
        return prompt

    def get_datetime(self):
        # เปลี่ยน Timezone เป็น Laos/Bangkok
        formatter = "%A, %m/%d/%Y, %H:%M:%S"
        now = datetime.now(timezone("Asia/Bangkok"))
        return now.strftime(formatter)

    def parse_messages(self, datas: List[Any]) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        for data in datas:
            if data["type"] == "human":
                messages.append(HumanMessage(content=data["data"]["content"]))
            elif data["type"] == "ai":
                messages.append(AIMessage(content=data["data"]["content"]))
            else:
                raise Exception("Message type not found.")
        return messages

    def get_base_history(self, session: dict[str, Any]):
        if "user_info" in session:
            base_history = {
                "type": "ai",
                "data": {
                    "content": f"Sabaidee {session['user_info']['name']}! Welcome to BCEL assistance. How can I help you today?"
                },
            }
            return base_history
        return BASE_HISTORY

    def get_config(self, uuid: str):
        return {
            "configurable": {
                "thread_id": uuid,
                "auth_token_getters": {
                    "my_google_service": lambda: self.get_user_id_token(uuid)
                },
                "checkpoint_ns": "",
            },
        }

    async def user_session_signout(self, uuid: str):
        checkpoint = empty_checkpoint()
        config = self.get_config(uuid)
        self._checkpointer.put(
            config=config, checkpoint=checkpoint, metadata={}, new_versions={}
        )
        del self._user_sessions[uuid]


# --- แก้ไข Prompt ให้เป็น BCEL Agent ---

PREFIX = """ເຈົ້າຄືຜູ້ຊ່ວຍອັດສະລິຍະຂອງ ທະນາຄານການຄ້າຕ່າງປະເທດລາວ ມະຫາຊົນ (BCEL).
ເປົ້າໝາຍຂອງເຈົ້າຄືການຊ່ວຍເຫຼືອລູກຄ້າໃນການໃຫ້ຂໍ້ມູນກ່ຽວກັບຜະລິດຕະພັນ ແລະ ການບໍລິການຕ່າງໆຂອງທະນາຄານ.

ໜ້າທີ່ຮັບຜິດຊອບຫຼັກ:
1. **ໃຫ້ຂໍ້ມູນຜະລິດຕະພັນ:** ອະທິບາຍລາຍລະອຽດກ່ຽວກັບ BCEL One, OnePay, i-Bank, ບັດ ATM/Credit, ແລະ ເຄື່ອງຮູດບັດ (EDC/POS) ໂດຍອີງຕາມຂໍ້ມູນທີ່ຄົ້ນຫາໄດ້ຈາກເຄື່ອງມື (Tools).
2. **ແນະນຳການບໍລິການ:** ອະທິບາຍວິທີການນຳໃຊ້, ຂັ້ນຕອນການຕິດຕັ້ງ, ແລະ ກຸ່ມເປົ້າໝາຍຂອງແຕ່ລະຜະລິດຕະພັນ.
3. **ການໃຊ້ພາສາ:** ເຈົ້າສາມາດສື່ສານໄດ້ຢ່າງຄ່ອງແຄ້ວທັງ **ພາສາລາວ** ແລະ **ພາສາອັງກິດ**.
   - ຖ້າລູກຄ້າຖາມເປັນພາສາລາວ, ໃຫ້ຕອບເປັນພາສາລາວ.
   - ຖ້າລູກຄ້າຖາມເປັນພາສາອັງກິດ, ໃຫ້ຕອບເປັນພາສາອັງກິດ.

ບຸກຄະລິກ ແລະ ນ້ຳສຽງ:
* ສຸພາບ, ເປັນມືອາຊີບ, ແລະ ເຕັມໃຈໃຫ້ບໍລິການ (ຄືກັບພະນັກງານທະນາຄານ).
* ໃຫ້ຂໍ້ມູນທີ່ກະທັດຮັດ, ຊັດເຈນ ແລະ ເຂົ້າໃຈງ່າຍ.
* ຖ້າບໍ່ຮູ້ຂໍ້ມູນ ຫຼື ຂໍ້ມູນບໍ່ມີໃນລະບົບ, ໃຫ້ແຈ້ງລູກຄ້າຢ່າງສຸພາບ ແລະ ແນະນຳໃຫ້ຕິດຕໍ່ສາຂາທະນາຄານ ຫຼື ເບິ່ງເວັບໄຊທ໌ BCEL.
* **ຫ້າມ** ແຕ່ງຂໍ້ມູນຂຶ້ນມາເອງ ຖ້າບໍ່ມີໃນຖານຂໍ້ມູນ.

ບໍລິບົດຂໍ້ມູນທີ່ມີ (Context):
ເຈົ້າສາມາດເຂົ້າເຖິງຖານຂໍ້ມູນຜະລິດຕະພັນຂອງ BCEL ຜ່ານເຄື່ອງມື `search_products`. ຂໍ້ມູນປະກອບມີ: ຊື່ຜະລິດຕະພັນ, ຄຳອະທິບາຍ, ປະເພດ, ສະຖານະ, ກຸ່ມລູກຄ້າເປົ້າໝາຍ, ແລະ ວິທີການຕິດຕັ້ງ.
"""

# 4. ແກ້ໄຂ SUFFIX ເປັນຄຳສັ່ງສັ້ນໆ
SUFFIX = """ເລີ່ມຕົ້ນໄດ້! ຈົ່ງໃຊ້ເຄື່ອງມື `search_products` ເພື່ອຊອກຫາຂໍ້ມູນຖ້າຈຳເປັນ. ຕອບກັບລູກຄ້າໂດຍກົງ."""