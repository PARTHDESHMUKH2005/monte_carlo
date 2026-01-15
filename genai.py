# gen_ai.py
from openai import OpenAI

client = OpenAI(api_key="s**")

def generate_risk_report(VaR_95, initial_capital, daily_inflow, daily_outflow, volatility):
    prompt = f"""
    You are a financial assistant. The 95% VaR is {VaR_95:.2f} 
    with initial capital {initial_capital}, daily inflow {daily_inflow}, 
    daily outflow {daily_outflow}, and volatility {volatility}.
    Explain the risk in simple language and suggest one action to reduce risk.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
