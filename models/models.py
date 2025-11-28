# Copyright 2023 Google LLC
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

import ast
import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

class Product(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 1. กำหนด Field 7 ตัว ให้ตรงກັບ CSV ของคุณเป๊ะๆ
    product_id: str
    product_name: str
    description: str
    type: str
    status: str
    audience: str
    products_types: str  # ใน CSV คุณใช้ P ตัวใหญ่และ s (Products_types)
    installation: str
    
    # 2. Field พิเศษสำหรับเก็บ Vector (Embedding)
    embedding: Optional[list[float]] = None

    # 3. ตัวช่วยแปลงค่า (Validator) เหมือนต้นฉบับ
    @field_validator("embedding", mode="before")
    def validate_embedding(cls, v):
        if isinstance(v, str):
            try:
                # พยายามแปลง String เป็น List
                v = ast.literal_eval(v)
                # แปลงไส้ในให้เป็น float ทุกตัว
                v = [float(f) for f in v]
            except (ValueError, SyntaxError):
                return None
        return v