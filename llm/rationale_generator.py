import os
import sys
import time
import json
import logging
import anthropic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RationaleGenerator:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Please check your .env file or environment variables.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def build_prompt(self, product_name, category, current_price, optimal_price, price_change_pct, expected_demand_change, elasticity, forecast_7d, competitor_weakness_score, sentiment_summary, margin, confidence):
        system_prompt = "You are a concise e-commerce pricing analyst. Write a 3-sentence pricing recommendation memo. Use specific numbers from the data. Do not use vague language like 'may' or 'could'. Lead with the recommended action."
        
        user_data = {
            "Product": product_name,
            "Category": category,
            "Current Price": f"${current_price:.2f}",
            "Recommended Optimal Price": f"${optimal_price:.2f}",
            "Price Change %": f"{price_change_pct:.1f}%",
            "Expected Demand Change Units": f"{expected_demand_change:.1f}",
            "Price Elasticity": f"{elasticity:.2f}",
            "7-Day Forecast (Units)": f"{forecast_7d:.1f}",
            "Competitor Weakness Score": f"{competitor_weakness_score:.2f}",
            "Sentiment Summary": sentiment_summary,
            "Expected Margin": f"{margin*100:.1f}%",
            "Model Confidence": confidence
        }
        
        user_prompt = f"```json\n{json.dumps(user_data, indent=2)}\n```\n\nBased on the above data, write the 3-sentence pricing recommendation memo for {product_name}."
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }

    def generate(self, prompt_inputs: dict):
        prompt_parts = self.build_prompt(**prompt_inputs)
        
        try:
            response = self.client.messages.create(
                model=LLM_MODEL,
                max_tokens=300,
                system=prompt_parts["system"],
                messages=[
                    {"role": "user", "content": prompt_parts["user"]}
                ]
            )
            
            return {
                "rationale": response.content[0].text,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "model": LLM_MODEL
            }
            
        except anthropic.RateLimitError:
            logging.warning("Rate limit hit. Waiting 5 seconds before retrying...")
            time.sleep(5)
            try:
                response = self.client.messages.create(
                    model=LLM_MODEL,
                    max_tokens=300,
                    system=prompt_parts["system"],
                    messages=[
                        {"role": "user", "content": prompt_parts["user"]}
                    ]
                )
                return {
                    "rationale": response.content[0].text,
                    "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                    "model": LLM_MODEL
                }
            except Exception as e:
                logging.error(f"Retry failed: {str(e)}")
                return {"rationale": "LLM unavailable — manual review required.", "tokens_used": 0, "error": str(e), "model": LLM_MODEL}
                
        except anthropic.APIError as e:
            logging.error(f"Anthropic API Error: {str(e)}")
            return {"rationale": "LLM unavailable — manual review required.", "tokens_used": 0, "error": str(e), "model": LLM_MODEL}
        except Exception as e:
            logging.error(f"Unexpected Error: {str(e)}")
            return {"rationale": "LLM unavailable — manual review required.", "tokens_used": 0, "error": str(e), "model": LLM_MODEL}

    def generate_batch(self, products_list: list):
        results = []
        for inputs in products_list:
            res = self.generate(inputs)
            results.append(res)
            time.sleep(1.0) # Respect rate limits
        return results
