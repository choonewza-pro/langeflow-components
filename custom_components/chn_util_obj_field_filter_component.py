from langflow.custom import Component
from langflow.io import Output, DataInput, MessageTextInput, DropdownInput
from langflow.schema import Data


class ObjectFieldFilter(Component):
    display_name = "Object Field Filter"
    description = "รับ Object เข้ามา แล้วเลือกได้ว่าจะ 'เก็บเฉพาะ field ที่ระบุ' หรือ 'ตัด field ที่ระบุออก'"
    icon = "filter"

    inputs = [
        DataInput(
            name="input_data",
            display_name="Input Object",
            info="ลากสายจาก component ที่ส่งออกเป็น Data object มาเชื่อมที่นี่",
        ),
        DropdownInput(
            name="mode",
            display_name="Mode",
            options=["Keep (เก็บเฉพาะ field ที่ระบุ)", "Reject (ตัด field ที่ระบุออก)"],
            value="Keep (เก็บเฉพาะ field ที่ระบุ)",
            info="เลือกโหมดการกรอง field",
        ),
        MessageTextInput(
            name="field_names",
            display_name="Field Names",
            info=(
                "ระบุชื่อ field ที่ต้องการ (รองรับ nested แบบ user.name) "
                "กด + เพื่อเพิ่มได้หลายรายการ"
            ),
            is_list=True,
        ),
    ]

    outputs = [
        Output(
            name="filtered_output",
            display_name="Filtered Object",
            method="filter_fields",
        ),
    ]

    def _get_nested(self, data: dict, field_path: str):
        """ดึงค่าจาก dict ตาม path แบบ nested เช่น 'user.profile.age' คืนค่า (found: bool, value)"""
        value = data
        for key in field_path.split("."):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return False, None
        return True, value

    def _set_nested(self, data: dict, field_path: str, value):
        """ตั้งค่าใน dict ตาม path แบบ nested โดยสร้าง dict ย่อยระหว่างทางถ้ายังไม่มี"""
        keys = field_path.split(".")
        current = data
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value

    def _delete_nested(self, data: dict, field_path: str):
        """ลบ key ออกจาก dict ตาม path แบบ nested (ถ้ามีอยู่จริง)"""
        keys = field_path.split(".")
        current = data
        for key in keys[:-1]:
            if not isinstance(current, dict) or key not in current:
                return
            current = current[key]
        if isinstance(current, dict) and keys[-1] in current:
            del current[keys[-1]]

    def filter_fields(self) -> Data:
        if self.input_data is None:
            raise ValueError("ไม่มี Input Object ต่อเข้ามา กรุณาลากสายจาก component ที่ส่งออกเป็น Data")

        source = self.input_data if isinstance(self.input_data, Data) else Data(data=self.input_data)
        original = source.data if isinstance(source.data, dict) else {}

        # field_names อาจเป็น list ของ string อยู่แล้วเพราะตั้ง is_list=True
        raw_fields = self.field_names or []
        field_list = [f.strip() for f in raw_fields if f and f.strip()]

        if self.mode.startswith("Keep"):
            # โหมด Keep: สร้าง object ใหม่ มีเฉพาะ field ที่ระบุ (ถ้ามีอยู่จริงใน original)
            result: dict = {}
            for field in field_list:
                found, value = self._get_nested(original, field)
                if found:
                    self._set_nested(result, field, value)
        else:
            # โหมด Reject: copy ของเดิมทั้งหมด แล้วลบ field ที่ระบุออก
            import copy

            result = copy.deepcopy(original)
            for field in field_list:
                self._delete_nested(result, field)

        output = Data(data=result)
        self.status = output
        return output