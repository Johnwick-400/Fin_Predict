from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import logging
import re
import os
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# API Keys and URLs
API_KEY = os.environ.get("MISTRAL_API_KEY", "l0rCW4iFL0WaRaPoWAgevw==UHn4QCEDnN74zp81")
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_NAME = "mistral-small"
API_NINJAS_KEY = 'l0rCW4iFL0WaRaPoWAgevw==UHn4QCEDnN74zp81'

def validate_response_logic(result, features):
    """
    Validates the logical consistency of the API response and corrects if needed
    """
    try:
        # Extract values for validation
        age = float(features['age'])
        monthly_income = float(features['income'])
        monthly_debt = float(features['existing_liabilities'])
        loan_amount = float(features['loan_amount'])
        loan_tenure = float(features['loan_tenure'])
        credit_score = float(features.get('credit_score', 650))
        loan_type = features.get('loan_type', 'personal').lower()
        
        # Calculate EMI
        annual_interest_rate = 0.10
        monthly_rate = annual_interest_rate / 12
        estimated_emi = (loan_amount * monthly_rate * (1 + monthly_rate)**loan_tenure) / ((1 + monthly_rate)**loan_tenure - 1)
        
        # Calculate DTI with new loan
        future_dti = (monthly_debt + estimated_emi) / monthly_income
        
        # Calculate loan-to-income ratio (annual)
        loan_to_annual_income = loan_amount / (monthly_income * 12)
        
        # Set criteria based on loan type
        if loan_type == 'education':
            min_age, max_age = 21, 35
            min_income = 15000
        elif loan_type == 'home':
            min_age, max_age = 21, 65
            min_income = 30000
        elif loan_type == 'vehicle':
            min_age, max_age = 21, 60
            min_income = 20000
        else:  # personal or default
            min_age, max_age = 21, 60
            min_income = 25000
        
        # Check for logical inconsistencies in the reasoning
        reasoning = result["detailed_reasoning"]
        
        # List of logical inconsistency patterns to check
        inconsistencies = [
            {
                "pattern": r"DTI ratio.*?([\d\.]+).*?exceeds.*?([\d\.]+)",
                "check": lambda match: float(match.group(1)) > float(match.group(2)),
                "fix": lambda match: reasoning.replace(
                    match.group(0), 
                    f"DTI ratio is {match.group(1)}, which is {'above' if float(match.group(1)) > float(match.group(2)) else 'below'} the recommended limit of {match.group(2)}"
                )
            },
            {
                "pattern": r"credit score.*?([\d\.]+).*?below.*?([\d\.]+)",
                "check": lambda match: float(match.group(1)) < float(match.group(2)),
                "fix": lambda match: reasoning.replace(
                    match.group(0), 
                    f"credit score is {match.group(1)}, which is {'below' if float(match.group(1)) < float(match.group(2)) else 'above'} the minimum requirement of {match.group(2)}"
                )
            },
            {
                "pattern": r"age.*?([\d\.]+).*?outside.*?([\d\.]+)-([\d\.]+)",
                "check": lambda match: not (float(match.group(2)) <= float(match.group(1)) <= float(match.group(3))),
                "fix": lambda match: reasoning.replace(
                    match.group(0), 
                    f"age ({match.group(1)}) is {'within' if float(match.group(2)) <= float(match.group(1)) <= float(match.group(3)) else 'outside'} the eligible range of {match.group(2)}-{match.group(3)}"
                )
            }
        ]
        
        # Check and fix inconsistencies
        corrected_reasoning = reasoning
        for inconsistency in inconsistencies:
            matches = re.finditer(inconsistency["pattern"], reasoning, re.IGNORECASE)
            for match in matches:
                if not inconsistency["check"](match):
                    corrected_reasoning = inconsistency["fix"](match)
        
        # Verify overall eligibility decision based on key factors
        key_eligibility_factors = {
            'age': min_age <= age <= max_age,
            'dti': future_dti <= 0.5,  # Standard 50% max DTI
            'credit': credit_score >= 650,  # Minimum acceptable credit score
            'income': monthly_income >= min_income
        }
        
        # Determine overall eligibility based on key factors
        should_be_eligible = all(key_eligibility_factors.values())
        
        # If there's a mismatch between our calculated eligibility and the API response
        if should_be_eligible != result["loan_eligibility"]:
            logger.warning(f"Eligibility mismatch detected. API: {result['loan_eligibility']}, Calculated: {should_be_eligible}")
            
            # If factors indicate eligibility but API says no
            if should_be_eligible and not result["loan_eligibility"]:
                corrected_reasoning = "Eligible for the requested loan. All critical criteria are met including age, income requirements, credit score, and debt-to-income ratio."
                result["loan_eligibility"] = True
                result["suggested_loan_amount"] = None
            
            # If factors indicate ineligibility but API says yes
            elif not should_be_eligible and result["loan_eligibility"]:
                failing_factors = [factor for factor, is_ok in key_eligibility_factors.items() if not is_ok]
                reasons = []
                
                for factor in failing_factors:
                    if factor == 'age':
                        reasons.append(f"Age ({int(age)}) is outside eligible range ({min_age}-{max_age})")
                    elif factor == 'dti':
                        reasons.append(f"Debt-to-income ratio ({future_dti:.2f}) exceeds maximum limit of 0.5")
                    elif factor == 'credit':
                        reasons.append(f"Credit score ({int(credit_score)}) is below minimum requirement of 650")
                    elif factor == 'income':
                        reasons.append(f"Monthly income (₹{monthly_income:,.0f}) is below minimum requirement (₹{min_income:,})")
                
                corrected_reasoning = f"Not eligible due to the following reasons: {'; '.join(reasons)}."
                result["loan_eligibility"] = False
                
                # Calculate suggested amount if DTI is the problem
                if 'dti' in failing_factors:
                    max_emi = (monthly_income * 0.5) - monthly_debt
                    if max_emi > 0:
                        suggested_amount = (max_emi * ((1 + monthly_rate)**loan_tenure - 1)) / (monthly_rate * (1 + monthly_rate)**loan_tenure)
                        result["suggested_loan_amount"] = round(suggested_amount)
        
        # Update the reasoning in the result
        if corrected_reasoning != reasoning:
            result["detailed_reasoning"] = corrected_reasoning
            
        return result
        
    except Exception as e:
        logger.error(f"Error in validate_response_logic: {str(e)}")
        # If validation fails, return the original result
        return result

def predict_with_mistral_ai(features):
    """
    Use Mistral AI API for loan prediction with enhanced prompt
    
    Args:
        features (dict): Dictionary containing loan application features
        
    Returns:
        dict: Prediction results with eligibility, confidence score, and reasoning
    """
    try:
        # Calculate important financial metrics to include in the prompt
        age = float(features['age'])
        monthly_income = float(features['income'])
        monthly_debt = float(features['existing_liabilities'])
        loan_amount = float(features['loan_amount'])
        loan_tenure = float(features['loan_tenure'])
        credit_score = float(features.get('credit_score', 650))
        loan_type = features.get('loan_type', 'personal').lower()
        
        # Calculate EMI using standard formula
        annual_interest_rate = 0.10  # 10% as a reasonable estimate for Indian loans
        monthly_rate = annual_interest_rate / 12
        estimated_emi = (loan_amount * monthly_rate * (1 + monthly_rate)**loan_tenure) / ((1 + monthly_rate)**loan_tenure - 1)
        
        # Calculate DTI (current and after loan)
        current_dti = monthly_debt / monthly_income
        future_dti = (monthly_debt + estimated_emi) / monthly_income
        
        # Calculate loan-to-income ratio (annual)
        loan_to_annual_income = loan_amount / (monthly_income * 12)
        
        # Format all the features for the prompt
        formatted_details = f"""
Age: {age} years
Monthly Income: ₹{monthly_income:,.2f}
Employment Status: {features['employment_status']}
Requested Loan Amount: ₹{loan_amount:,.2f}
Loan Tenure: {loan_tenure} months
Credit Score: {credit_score}
Existing Monthly Liabilities: ₹{monthly_debt:,.2f}
Loan Type: {features['loan_type']}

CALCULATED METRICS:
Estimated Monthly EMI: ₹{estimated_emi:,.2f}
Current Debt-to-Income Ratio: {current_dti:.2f} ({current_dti*100:.1f}%)
Future Debt-to-Income Ratio (including new loan EMI): {future_dti:.2f} ({future_dti*100:.1f}%)
Loan-to-Annual-Income Ratio: {loan_to_annual_income:.2f} ({loan_to_annual_income*100:.1f}%)
"""
        
        # Create a more precise prompt with strict rules
        prompt = f"""
You are an expert loan officer at an Indian bank specializing in {features['loan_type']} loans. Your task is to evaluate a loan application based on Indian banking standards.

LOAN APPLICATION DETAILS:
{formatted_details}

STRICT INDIAN BANKING CRITERIA - The applicant MUST meet ALL of these criteria to be eligible:
1. Age must be between:
   - 21-60 years for personal loans
   - 21-35 years for education loans
   - 21-65 years for home loans
   - 21-60 years for vehicle loans

2. Debt-to-Income Ratio (DTI) including the new loan EMI must be LESS THAN OR EQUAL TO 0.5 (50%)

3. Credit Score must be:
   - At least 650 (minimum requirement)
   - Above 750 is considered good

4. Minimum monthly income must be at least:
   - Personal loans: ₹25,000/month
   - Education loans: ₹15,000/month
   - Home loans: ₹30,000/month
   - Vehicle loans: ₹20,000/month

5. Loan-to-income ratio should not exceed recommended limits based on loan type

IMPORTANT: When giving your assessment, make sure your reasoning is LOGICALLY CONSISTENT. If you say a number exceeds a limit, double-check that it actually does. If you say a value is below a minimum, verify that mathematically.

Based on these criteria, provide your assessment in the following JSON format:
{{
    "loan_eligibility": true or false (must be based strictly on meeting ALL criteria above),
    "confidence_score": A number between 1-100 representing your confidence,
    "detailed_reasoning": "A clear explanation with accurate key factors and correct mathematical comparisons",
    "suggested_loan_amount": If not eligible for requested amount but could qualify for a lower amount, suggest it here, otherwise null
}}

Return ONLY the JSON response with no additional text.
"""
        
        # Prepare the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are an expert loan officer in India who strictly follows banking regulations and provides mathematically accurate assessments."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # Low temperature for more consistent outputs
            "max_tokens": 500
        }
        
        logger.info("Sending request to Mistral AI API")
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"API error: {response.status_code}, {response.text}")
            raise Exception(f"API request failed with status code: {response.status_code}")
            
        # Extract the response
        api_response = response.json()
        content = api_response["choices"][0]["message"]["content"]
        
        logger.info(f"Raw API response: {content}")
        
        # Try to parse the JSON response
        try:
            # The API might return the JSON with text before or after, so we need to extract it
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            result = json.loads(content)
            
            # Validate the result format
            required_fields = ["loan_eligibility", "confidence_score", "detailed_reasoning"]
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Missing field in API response: {field}")
                    # Set default values for missing fields
                    if field == "loan_eligibility":
                        result[field] = False
                    elif field == "confidence_score":
                        result[field] = 50.0
                    elif field == "detailed_reasoning":
                        result[field] = "Unable to provide detailed reasoning."
            
            # Ensure suggested_loan_amount is present
            if "suggested_loan_amount" not in result:
                result["suggested_loan_amount"] = None
                
            # Ensure the confidence score is a number between 1-100
            try:
                result["confidence_score"] = float(result["confidence_score"])
                result["confidence_score"] = max(1, min(100, result["confidence_score"]))
            except (ValueError, TypeError):
                result["confidence_score"] = 50.0
            
            # Validate and potentially correct logical inconsistencies
            validated_result = validate_response_logic(result, features)
            return validated_result
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {content}")
            # Provide a fallback response based on basic rules
            return fallback_loan_prediction(features)
            
    except Exception as e:
        logger.error(f"Error in predict_with_mistral_ai: {str(e)}")
        # Use fallback method if API fails
        return fallback_loan_prediction(features)

def fallback_loan_prediction(features):
    """
    Fallback method using simple rules in case the API fails
    """
    try:
        # Extract and convert features
        age = float(features['age'])
        monthly_income = float(features['income'])
        monthly_debt = float(features['existing_liabilities'])
        loan_amount = float(features['loan_amount'])
        loan_tenure = float(features['loan_tenure'])
        credit_score = float(features.get('credit_score', 650))
        loan_type = features.get('loan_type', 'personal').lower()
        
        # Calculate EMI
        annual_interest_rate = 0.10
        monthly_rate = annual_interest_rate / 12
        estimated_emi = (loan_amount * monthly_rate * (1 + monthly_rate)**loan_tenure) / ((1 + monthly_rate)**loan_tenure - 1)
        
        # Calculate DTI with new loan
        future_dti = (monthly_debt + estimated_emi) / monthly_income
        
        # Set criteria based on loan type
        if loan_type == 'education':
            min_age, max_age = 19, 35
            min_income = 15000
        elif loan_type == 'home':
            min_age, max_age = 21, 65
            min_income = 30000
        elif loan_type == 'vehicle':
            min_age, max_age = 21, 60
            min_income = 20000
        else:  # personal or default
            min_age, max_age = 21, 60
            min_income = 25000
            
        # Check eligibility criteria
        is_age_ok = min_age <= age <= max_age
        is_credit_ok = credit_score >= 650
        is_dti_ok = future_dti <= 0.5
        is_income_ok = monthly_income >= min_income
        
        reasons = []
        if not is_age_ok:
            reasons.append(f"Age ({int(age)}) outside eligible range ({min_age}-{max_age})")
        if not is_credit_ok:
            reasons.append(f"Credit score ({int(credit_score)}) below minimum requirement (650)")
        if not is_dti_ok:
            reasons.append(f"DTI ratio ({future_dti:.2f}) exceeds maximum (0.5)")
        if not is_income_ok:
            reasons.append(f"Income (₹{monthly_income:,.0f}) below minimum (₹{min_income:,})")
            
        is_eligible = is_age_ok and is_credit_ok and is_dti_ok and is_income_ok
        
        # Calculate suggested amount if not eligible due to DTI
        suggested_amount = None
        if not is_eligible and not is_dti_ok:
            max_emi = (monthly_income * 0.5) - monthly_debt
            if max_emi > 0:
                suggested_amount = (max_emi * ((1 + monthly_rate)**loan_tenure - 1)) / (monthly_rate * (1 + monthly_rate)**loan_tenure)
                suggested_amount = max(0, min(loan_amount * 0.8, suggested_amount))
        
        message = "Eligible for the requested loan." if is_eligible else f"Not eligible due to: {', '.join(reasons)}."
        
        return {
            "loan_eligibility": is_eligible,
            "confidence_score": 70.0,  # Conservative confidence for fallback
            "detailed_reasoning": message,
            "suggested_loan_amount": round(suggested_amount) if suggested_amount else None
        }
        
    except Exception as e:
        logger.error(f"Error in fallback_loan_prediction: {str(e)}")
        # Ultra fallback with generic response
        return {
            "loan_eligibility": False,
            "confidence_score": 50.0,
            "detailed_reasoning": "Unable to process loan application due to technical issues.",
            "suggested_loan_amount": None
        }

def get_interest_rates_from_api_ninjas():
    """Fetch current interest rates for India from API Ninjas"""
    try:
        api_url = 'https://api.api-ninjas.com/v1/interestrate?country=India'
        headers = {'X-Api-Key': API_NINJAS_KEY}
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 200:
            logger.info("Successfully fetched interest rates from API Ninjas")
            return response.json()
        else:
            logger.error(f"API Ninjas error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error fetching interest rates from API Ninjas: {str(e)}")
        return None

def get_indian_banks_offers(loan_type: str):
    """
    Get bank offers enhanced with real-time interest rate data if available
    """
    interest_data = get_interest_rates_from_api_ninjas()
    base_rates = {
        "education": 9.5,
        "personal": 10.5,
        "home": 8.5,
        "vehicle": 9.0
    }
    # Robust check for central bank rates
    central_rates = interest_data.get('central_bank_rates') if interest_data else None
    current_rate = None
    if (
        central_rates
        and isinstance(central_rates, list)
        and len(central_rates) > 0
        and isinstance(central_rates[0], dict)
        and 'rate_pct' in central_rates[0]
        and isinstance(central_rates[0]['rate_pct'], (int, float))
    ):
        current_rate = central_rates[0]['rate_pct']
        for loan in base_rates:
            base_rates[loan] = current_rate + (4.0 if loan == "personal" else 3.0 if loan == "education" else 2.0 if loan == "vehicle" else 1.5)
    banks = [
        {
            "bank_name": "State Bank of India",
            "interest_rate": base_rates['education'] - 0.3 if loan_type == "education" else base_rates['personal'] if loan_type == "personal" else base_rates['home'] if loan_type == "home" else base_rates['vehicle'],
            "min_income": 15000 if loan_type == "education" else 25000,
            "min_credit_score": 650,
            "max_loan_amount": 2000000,
            "processing_fee": "₹2,000",
            "special_note": "No prepayment penalty"
        },
        {
            "bank_name": "HDFC Bank",
            "interest_rate": base_rates['education'] if loan_type == "education" else base_rates['personal'] + 0.5 if loan_type == "personal" else base_rates['home'] + 0.2 if loan_type == "home" else base_rates['vehicle'] + 0.2,
            "min_income": 18000 if loan_type == "education" else 27000,
            "min_credit_score": 700,
            "max_loan_amount": 2500000,
            "processing_fee": "₹2,500",
            "special_note": "Quick disbursal"
        },
        {
            "bank_name": "ICICI Bank",
            "interest_rate": base_rates['education'] + 0.3 if loan_type == "education" else base_rates['personal'] + 1.0 if loan_type == "personal" else base_rates['home'] + 0.4 if loan_type == "home" else base_rates['vehicle'] + 0.4,
            "min_income": 16000 if loan_type == "education" else 26000,
            "min_credit_score": 675,
            "max_loan_amount": 2200000,
            "processing_fee": "₹2,000",
            "special_note": "Flexible tenure"
        },
        {
            "bank_name": "Axis Bank",
            "interest_rate": base_rates['education'] + 0.5 if loan_type == "education" else base_rates['personal'] + 1.5 if loan_type == "personal" else base_rates['home'] + 0.5 if loan_type == "home" else base_rates['vehicle'] + 0.6,
            "min_income": 17000 if loan_type == "education" else 28000,
            "min_credit_score": 680,
            "max_loan_amount": 2100000,
            "processing_fee": "₹1,500",
            "special_note": "Minimal documentation"
        }
    ]
    banks_sorted = sorted(banks, key=lambda x: x["interest_rate"])
    return banks_sorted

def scrape_personal_loan_rates():
    url = "https://www.bankbazaar.com/personal-loan-interest-rates.html"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "tbl"})
        offers = []
        if table:
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) >= 4:
                    bank = cols[0].get_text(strip=True)
                    rate = cols[1].get_text(strip=True).replace("%", "")
                    min_income = cols[2].get_text(strip=True)
                    processing_fee = cols[3].get_text(strip=True)
                    logger.info(f"Scraped bank: {bank}, rate: {rate}, min_income: {min_income}, fee: {processing_fee}")
                    bank_normalized = bank.lower().replace(" ", "")
                    if bank_normalized in ["statebankofindia", "hdfcbank", "icicibank", "axisbank"]:
                        offers.append({
                            "bank_name": bank,
                            "interest_rate": float(rate.split('-')[0]),
                            "min_income": min_income,
                            "min_credit_score": 650,
                            "max_loan_amount": None,
                            "processing_fee": processing_fee,
                            "special_note": ""
                        })
        logger.info(f"Total offers scraped: {len(offers)}")
        offers = sorted(offers, key=lambda x: x["interest_rate"])
        return offers
    except Exception as e:
        logger.error(f"Error scraping loan rates: {str(e)}")
        return []

@app.route('/api/predict-eligibility', methods=['POST'])
def predict_eligibility():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ['age', 'income', 'employmentStatus', 'loanAmount', 'loanTenure', 'existingLiabilities']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        features = {
            'age': float(data['age']),
            'income': float(data['income']),
            'employment_status': data['employmentStatus'],
            'loan_amount': float(data['loanAmount']),
            'loan_tenure': float(data['loanTenure']),
            'credit_score': float(data.get('creditScore', 650)),
            'existing_liabilities': float(data['existingLiabilities']),
            'loan_type': data.get('loanType', 'personal')
        }

        logger.info(f"Received loan eligibility request: {features}")

        # Get prediction from Mistral AI or fallback
        result = predict_with_mistral_ai(features)
        logger.info(f"Final prediction result: {result}")

        response = {
            'isEligible': result['loan_eligibility'],
            'confidence': result['confidence_score'],
            'message': result['detailed_reasoning'],
            'suggestedAmount': result['suggested_loan_amount']
        }

        # Get bank offers (scrape for personal, static for others)
        if features.get('loan_type', 'personal') == 'personal':
            bank_offers = scrape_personal_loan_rates()
            if not bank_offers:
                logger.warning("Scraping failed or returned no offers, using static offers.")
                bank_offers = get_indian_banks_offers('personal')
        else:
            bank_offers = get_indian_banks_offers(features.get('loan_type', 'personal'))

        response["bankOffers"] = bank_offers
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in predict_eligibility: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
