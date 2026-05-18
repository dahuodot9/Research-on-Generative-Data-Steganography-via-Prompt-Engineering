# test_api.py
"""快速测试文心API是否连通"""
from generate_data import call_wenxin
from config import API_CONFIGS

print("=" * 50)
print("测试 ERNIE-Lite-Pro-128K 连通性 (新版接口)")
print("=" * 50)

config = API_CONFIGS["wenxin"]
print(f"\n密钥前缀: {config['api_key'][:20]}...")
print(f"接口域名: qianfan.baidubce.com/v2")
print("\n发送测试请求...")

result = call_wenxin("请用一句话介绍人工智能", config)

if result:
    print(f"\n✅ 连接成功！")
    print(f"返回内容: {result[:150]}...")
else:
    print(f"\n❌ 连接失败！请检查:")
    print(f"  1. api_key 格式是否为 bce-v3/ALTAK-xxx")
    print(f"  2. 该密钥是否已开通 ERNIE-Lite-Pro-128K 调用权限")
    print(f"  3. 模型名是否填写正确: ernie-lite-pro-128k")