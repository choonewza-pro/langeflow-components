import json
from langflow.custom import Component
from langflow.io import Output, DataInput, MessageTextInput
from langflow.schema.message import Message
from langflow.schema import Data


class TextObjectConcat(Component):
    display_name = "Text + Object Concat"
    description = "รับ Object และข้อความที่พิมพ์ แล้วต่อกันเป็น text รูปแบบ 'text_input {json_text}'"
    icon = "combine"

    inputs = [
        MessageTextInput(
            name="text_input",
            display_name="Text Input",
            info="ข้อความที่ต้องการนำไปต่อหน้า JSON",
        ),
        DataInput(
            name="input_data",
            display_name="Input Object",
            info="ลากสายจาก component ที่ส่งออกเป็น Data object มาเชื่อมที่นี่",
        ),
    ]

    outputs = [
        Output(
            name="output_text",
            display_name="Output Text",
            method="concat_text",
        ),
    ]

    def concat_text(self) -> Message:
        if self.input_data is None:
            raise ValueError("ไม่มี Input Object ต่อเข้ามา กรุณาลากสายจาก component ที่ส่งออกเป็น Data")

        source = self.input_data if isinstance(self.input_data, Data) else Data(data=self.input_data)
        obj = source.data if isinstance(source.data, dict) else {}

        # แปลง object เป็น JSON string (ensure_ascii=False เพื่อให้ตัวอักษรไทยไม่กลายเป็น \uXXXX)
        json_text = json.dumps(obj, ensure_ascii=False)

        text = self.text_input or ""
        result_text = f"{text} {json_text}"

        message = Message(text=result_text)
        self.status = message
        return message