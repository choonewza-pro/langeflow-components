from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import MessageInput
from lfx.io import Output
from lfx.schema.data import Data
from lfx.schema.message import Message


class ImageCheckComponent(Component):
    display_name = "Image Check"
    description = (
        "เช็คว่า Message ที่ต่อมาจาก Chat Input มีไฟล์รูปภาพแนบมาหรือไม่ "
        "ถ้ามี -> ส่ง Message เดิมต่อไปทาง Success ถ้าไม่มีรูปเลยสักรูป -> ส่ง error ทาง Error"
    )
    icon = "ImageOff"
    name = "ImageCheck"

    inputs = [
        MessageInput(
            name="input_value",
            display_name="Input",
            info="ข้อความ + รูปภาพที่ผู้ใช้ส่งมาทาง Chat Input",
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Success", name="success", method="run_success", group_outputs=True),
        Output(display_name="Error", name="error", method="run_error", group_outputs=True),
    ]

    def _has_image(self) -> bool:
        """เช็คว่า Message ที่ได้จาก Chat Input มีไฟล์รูปแนบมาไหม (เช็คแค่ 'มี/ไม่มี' เท่านั้น).

        ไม่พยายาม resolve หรือเช็คว่าไฟล์มีอยู่จริงบน disk เอง เพราะ path ที่เก็บใน
        Message.files เป็น storage path ภายในของ Langflow (รูปแบบ FLOW_ID/TIMESTAMP_FILENAME.EXT)
        ไม่ใช่ OS path ตรงๆ — การ resolve ที่ถูกต้องทำผ่าน Message.to_lc_message() เท่านั้น
        """
        return bool(getattr(self.input_value, "files", None))

    def _get_result(self) -> tuple[Message | None, str | None]:
        """คืนค่า (message, error_message) — สำเร็จ error_message จะเป็น None.

        cache ไว้ใน self.ctx เพื่อไม่ให้เช็คซ้ำ ในกรณีที่ทั้ง Success และ Error
        output ถูกต่อไปใช้งานพร้อมกัน (ทั้งสอง method จะถูกเรียกโดย Langflow)
        """
        if "image_check_result" in self.ctx:
            return self.ctx["image_check_result"]

        if not self._has_image():
            error_message = "ไม่พบรูปภาพที่ผู้ใช้ส่งมาทาง Chat Input กรุณาแนบรูปภาพก่อนส่งข้อความ"
            result: tuple[Message | None, str | None] = (None, error_message)
            self.ctx["image_check_result"] = result
            self.status = error_message
            return result

        result = (self.input_value, None)
        self.ctx["image_check_result"] = result
        self.status = self.input_value
        return result

    def run_success(self) -> Message:
        message, error = self._get_result()
        if error is not None:
            self.status = f"ล้มเหลว: {error}"
            self.stop("success")
            return Message(text="")
        return message

    def run_error(self) -> Data:
        message, error = self._get_result()
        if error is None:
            self.stop("error")
            return Data(data={})

        error_data = Data(data={"error": error, "error_type": "missing_image"})
        self.status = error_data
        return error_data
