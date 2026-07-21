import json
from langflow.custom import Component
from langflow.io import Output, DataInput, MessageTextInput
from langflow.schema.message import Message
from langflow.schema import Data


class TextObjectConcat(Component):
    display_name = "Text + Object Concat"
    description = "รับ Object และข้อความที่พิมพ์ แล้วต่อกันเป็น text รูปแบบ 'text_input {json_text}' พร้อมแนบรูปภาพต้นฉบับกลับไปด้วย (ถ้ามี)"
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

        # ★ ดึงไฟล์รูปภาพต้นฉบับจาก self.ctx ที่ ImageCheck เก็บไว้
        # (ImageCheck._get_result() เก็บ tuple (message, error) ไว้ที่ self.ctx["image_check_result"])
        # ไม่ต้องลากสาย edge ตรงมาจาก ImageCheck เลย เพราะ self.ctx เป็น context ที่ใช้ร่วมกัน
        # ได้ทั้ง flow อยู่แล้ว วิธีนี้เลี่ยงปัญหา fan-out จาก component ที่ใช้
        # group_outputs + self.stop() ซึ่งทำให้ Langflow build ลำดับการรันผิดพลาด
        # (error "Input data cannot be None" ที่เจอก่อนหน้านี้)
        cached = self.ctx.get("image_check_result")
        if cached:
            original_message, error = cached
            if original_message is not None:
                files = getattr(original_message, "files", None)
                if files:
                    message.files = files

        self.status = message
        return message