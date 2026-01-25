from smolagents import TransformersModel, Tool, CodeAgent
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    return parser.parse_args()


class tool_tester(Tool):
    name = "thai_to_thai_gloss_translator"
    description = (
        "MANDATORY tool. ALWAYS call this tool for Thai → Thai-gloss. "
        "Never answer directly. Just used it once and return answer you think correct"
    )
    inputs = {
        "task": {
            "type": "string",
            "description": "machine translation thai to thai gloss"
        }
    }
    output_type = "string"

    def forward(self, task: str) -> str:
        print("Calling tool success")
        return "ไม่รู้"



if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = TransformersModel(
        model_id=args.model_name_or_path,
        max_new_tokens=4096,
        device_map='auto',
    )

    agent = CodeAgent(tools=[tool_tester()], model=model, add_base_tools=False)
    result = agent.run(
        "ภาคกลาง อุณหภูมิต่ำสุด 24 องศา สูงสุด 36 องศา มีฝนฟ้าคะนองร้อยละ 60 ของพื้นที่ค่ะ ส่วนมากจะตกทางด้านตะวันตกของภาค"
    )
    print(result)