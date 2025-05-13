import requests
import json
import re
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import functools
import time
from datetime import datetime, timedelta
import flask_cors
import concurrent.futures
import threading
import hashlib  # For better cache key generation

# Load environment variables
load_dotenv()

app = Flask(__name__)
flask_cors.CORS(app)

# Environment-based configuration
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
DEFAULT_MAX_TOKENS = int(os.getenv('DEFAULT_MAX_TOKENS', 800))  # Increased token count
REQUEST_TIMEOUT = None  # Disable timeout completely
FAST_MODEL = os.getenv('FAST_MODEL', OLLAMA_MODEL)

# Enhanced cache setup
CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
CACHE_EXPIRATION = int(os.getenv('CACHE_EXPIRATION', 3600))
cache = {}
cache_lock = threading.Lock()

# Rate limiting
RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'false').lower() == 'true'
RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', 10))
rate_limit_store = {}
rate_limit_lock = threading.Lock()

# Increase number of parallel workers
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 10))  # Increased from 5 to 10

# Maximum resume length to process
MAX_RESUME_LENGTH = int(os.getenv('MAX_RESUME_LENGTH', 2000))  # Shortened from 3000

# Precompiled regex patterns for better performance
PREFIX_PATTERNS = [
    re.compile(r'^(Enhanced|Paraphrased|Rewritten|Modern) (text|version|resume):\s*', re.IGNORECASE),
    re.compile(r'^Here is the (rewritten|enhanced) (text|resume):\s*', re.IGNORECASE),
    re.compile(r'^Here\'s the [^:]*:\s*', re.IGNORECASE),
    re.compile(r'^Below is [^:]*:\s*', re.IGNORECASE),
    re.compile(r'^I\'ve (rewritten|enhanced|paraphrased) [^:]*:\s*', re.IGNORECASE)
]
MARKDOWN_PATTERN = re.compile(r'```[a-z]*\n|```')
NEWLINE_PATTERN = re.compile(r'\n{3,}')

def cache_key(func_name, text):
    """Generate a more efficient cache key using MD5 hash"""
    # Only hash first 100 chars for speed, but use MD5 for better distribution
    text_sample = text[:100]
    return f"{func_name}:{hashlib.md5(text_sample.encode()).hexdigest()}"

def cache_result(func):
    """Decorator to cache function results with improved key generation"""
    @functools.wraps(func)
    def wrapper(text, *args, **kwargs):
        if not CACHE_ENABLED:
            return func(text, *args, **kwargs)
            
        # Create a cache key from function name and text sample
        key = cache_key(func.__name__, text)
        
        # Thread-safe cache check
        with cache_lock:
            if key in cache:
                result, timestamp = cache[key]
                if datetime.now() - timestamp < timedelta(seconds=CACHE_EXPIRATION):
                    return result
        
        # Call the function and cache the result
        result = func(text, *args, **kwargs)
        
        # Thread-safe cache update
        with cache_lock:
            cache[key] = (result, datetime.now())
        
        return result
    return wrapper

def check_rate_limit(ip_address):
    """Check if request exceeds rate limit"""
    if not RATE_LIMIT_ENABLED:
        return True
    
    current_time = datetime.now()
    minute_ago = current_time - timedelta(minutes=1)
    
    with rate_limit_lock:
        if ip_address not in rate_limit_store:
            rate_limit_store[ip_address] = []
        
        # Remove requests older than 1 minute
        rate_limit_store[ip_address] = [t for t in rate_limit_store[ip_address] if t > minute_ago]
        
        if len(rate_limit_store[ip_address]) < RATE_LIMIT_PER_MINUTE:
            rate_limit_store[ip_address].append(current_time)
            return True
        
        return False

# Health check endpoint for Flutter app
@app.route('/health', methods=['GET'])
def health_check():
    # Check if Ollama is running and responsive
    ollama_status = "unknown"
    model_loaded = False
    
    try:
        # Try a simple query to see if Ollama is responsive
        response = requests.get(OLLAMA_API_URL.replace('/api/generate', '/api/tags'), timeout=5)  # Keep a small timeout for health check
        if response.status_code == 200:
            ollama_status = "running"
            
            # Check if our model is loaded
            models = response.json().get('models', [])
            for model in models:
                if model.get('name', '') == OLLAMA_MODEL:
                    model_loaded = True
                    break
        else:
            ollama_status = f"error: {response.status_code}"
    except Exception as e:
        ollama_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "ok" if ollama_status == "running" else "degraded",
        "message": "Service is up and running",
        "details": {
            "ollama_status": ollama_status,
            "model_loaded": model_loaded,
            "cache_entries": len(cache),
            "default_model": OLLAMA_MODEL,
            "fast_model": FAST_MODEL,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "max_resume_length": MAX_RESUME_LENGTH,
            "timeout_disabled": REQUEST_TIMEOUT is None
        }
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to Resume Paraphrasing API", 
                    "endpoints": {
                        "/health": "GET - Check service status",
                        "/paraphrase": "POST - Paraphrase resume text"
                    }})

@app.route('/paraphrase', methods=['POST'])
def paraphrase_resume():
    # Check rate limit
    client_ip = request.remote_addr
    if not check_rate_limit(client_ip):
        return jsonify({"error": "Rate limit exceeded. Try again later."}), 429
    
    data = request.get_json()
    resume_text = data.get('text', '')
    
    if not resume_text:
        return jsonify({"error": "No resume text provided"}), 400
    
    # Log initial request details
    print(f"Processing resume with {len(resume_text)} characters")
    
    # Truncate very long resume texts to improve performance
    original_length = len(resume_text)
    if len(resume_text) > MAX_RESUME_LENGTH:
        resume_text = resume_text[:MAX_RESUME_LENGTH] + "..."
        was_truncated = True
    else:
        was_truncated = False
    
    # Detect resume sections for better context-aware prompting
    sections = detect_resume_sections(resume_text)
    print(f"Detected sections: {sections}")
    
    # Define paraphrase options with improved prompts
    paraphrase_options = [
        {
            "function": specialization_emphasis_paraphrase,
            "type": "SpecializationEmphasis",
            "description": "Highlights your expertise and specialized skills"
        },
        {
            "function": modern_concise_paraphrase,
            "type": "ModernConcise", 
            "description": "Contemporary, streamlined language for modern recruiters"
        },
        {
            "function": role_highlight_paraphrase,
            "type": "RoleHighlight",
            "description": "Emphasizes positions and responsibilities"
        },
        {
            "function": technical_distinction_paraphrase,
            "type": "TechnicalDistinction",
            "description": "Focuses on technical skills and achievements"
        },
        {
            "function": descriptive_detailed_paraphrase,
            "type": "DescriptiveDetailed",
            "description": "Comprehensive, detailed presentation of experience"
        }
    ]
    
    # Execute paraphrasing functions in parallel with improved error handling
    results = []
    failures = 0
    
    # Process options in sequence rather than parallel if resume is large
    if len(resume_text) > 2000:
        # For long resumes, process sequentially to avoid overwhelming the model
        for option in paraphrase_options[:3]:  # Only try 3 options for large resumes
            try:
                result = option["function"](resume_text, sections=sections)
                if result:
                    results.append({
                        "type": option["type"],
                        "text": result,
                        "description": option["description"]
                    })
                    # If we have at least one good result, we can proceed
                    if len(results) >= 1:
                        break
            except Exception as e:
                print(f"Error in sequential processing {option['type']}: {str(e)}")
                failures += 1
    else:
        # For shorter resumes, use parallel processing with a reduced worker count
        # Modified to not use timeout in as_completed, since we've disabled timeouts in the requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, MAX_WORKERS)) as executor:
            # Submit all tasks
            future_to_option = {
                executor.submit(
                    option["function"], 
                    resume_text, 
                    sections=sections
                ): option 
                for option in paraphrase_options
            }
            
            # Collect results without timeout
            for future in concurrent.futures.as_completed(future_to_option):
                option = future_to_option[future]
                try:
                    result = future.result()
                    if result:
                        results.append({
                            "type": option["type"],
                            "text": result,
                            "description": option["description"]
                        })
                except Exception as e:
                    print(f"Error in parallel processing {option['type']}: {str(e)}")
                    failures += 1
    
    # If we don't have at least 2 results, try the fallback mechanism
    if len(results) < 2:
        print(f"Insufficient results ({len(results)}), trying fallback")
        fallback_result = ensure_paraphrase_results(resume_text, sections)
        if fallback_result:
            # Check if this result type is already in our results
            existing_types = [r["type"] for r in results]
            if fallback_result["type"] not in existing_types:
                results.append(fallback_result)
    
    # If we still have no results, add the original text as a last resort
    if not results:
        print("All paraphrasing attempts failed, returning original")
        results.append({
            "type": "Original",
            "text": resume_text,
            "description": "Original resume text (paraphrasing unavailable)"
        })
    
    # Log completion information
    print(f"Completed processing with {len(results)} successful paraphrases and {failures} failures")
    
    return jsonify({
        "original": resume_text,
        "paraphrases": results,
        "meta": {
            "original_length": original_length,
            "truncated": was_truncated,
            "sections_detected": sections,
            "success_rate": f"{len(results)}/{len(paraphrase_options) + 1}"
        }
    })

# Improved paraphrasing functions with more specific instructions

@cache_result
def specialization_emphasis_paraphrase(text, sections=None):
    """Emphasis in specialization: Highlights expertise and specialized skills"""
    prompt = f"""You are a professional resume writer. Rewrite this resume to emphasize specialized skills and expertise.
    Maintain all original information but use stronger wording to highlight technical specialization.
    
    Resume:
    {text}
    
    Rewritten resume (with specialized skills emphasized):"""
    
    response = query_llama(prompt, 0.2, DEFAULT_MAX_TOKENS)
    return clean_llama_response(response) or None

@cache_result
def modern_concise_paraphrase(text, sections=None):
    """Modern and concise: Contemporary, streamlined language for modern recruiters"""
    prompt = f"""As a professional resume writer, make this resume more concise and modern.
    Use contemporary language and strong action verbs. Keep all key information.
    
    Resume:
    {text}
    
    Modern, concise version:"""
    
    response = query_llama(prompt, 0.2, DEFAULT_MAX_TOKENS)
    return clean_llama_response(response) or None

@cache_result
def role_highlight_paraphrase(text, sections=None):
    """Highlighting roles: Emphasizes positions and responsibilities"""
    prompt = f"""Rewrite this resume to emphasize job titles, roles, and key responsibilities.
    Make position titles and roles stand out prominently.
    
    Resume:
    {text}
    
    Role-emphasized version:"""
    
    response = query_llama(prompt, 0.2, DEFAULT_MAX_TOKENS)
    return clean_llama_response(response) or None

@cache_result
def technical_distinction_paraphrase(text, sections=None):
    """Technical distinction: Focuses on technical skills and achievements"""
    prompt = f"""Rewrite this resume emphasizing technical skills, tools, and technological achievements.
    Focus on technical terminology and quantifiable technical accomplishments.
    
    Resume:
    {text}
    
    Technically-focused version:"""
    
    response = query_llama(prompt, 0.2, DEFAULT_MAX_TOKENS)
    return clean_llama_response(response) or None

@cache_result
def descriptive_detailed_paraphrase(text, sections=None):
    """Descriptive and detailed: Comprehensive, detailed presentation of experience"""
    prompt = f"""Rewrite this resume with more descriptive language highlighting accomplishments and impact.
    Use detailed, specific language to convey experience and achievements.
    
    Resume:
    {text}
    
    Descriptive, detailed version:"""
    
    response = query_llama(prompt, 0.25, DEFAULT_MAX_TOKENS)
    return clean_llama_response(response) or None
    
# Optimized Llama integration with removed timeout
def query_llama(prompt, temperature=0.3, max_tokens=None, model=None, retries=2):
    """Query the local Llama model via Ollama API with improved error handling and no timeout"""
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    
    if model is None:
        model = OLLAMA_MODEL
    
    # Add detailed logging for debugging
    print(f"Querying model: {model} with {max_tokens} tokens, temp={temperature}")
    
    for attempt in range(retries + 1):
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Removed timeout parameter to allow for unlimited response time
            response = requests.post(OLLAMA_API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                print(f"Received response: {len(response_text)} characters")
                return response_text
            else:
                print(f"Ollama API error: {response.status_code}, Response: {response.text}, Attempt {attempt+1}/{retries+1}")
                if attempt < retries:
                    time.sleep(2)  # Increased delay before retry
                    continue
                return None
        except Exception as e:
            print(f"Error querying Llama model: {str(e)}, Attempt {attempt+1}/{retries+1}")
            if attempt < retries:
                time.sleep(2)
                continue
            return None
    
    return None

def clean_llama_response(response):
    """Improved cleaning function for LLaMA responses"""
    if not response:
        return None
    
    # Print the raw response for debugging
    print(f"Raw response length: {len(response)}")
    if len(response) < 50:
        print(f"Short response: {response}")
    
    # Strip leading/trailing whitespace
    response = response.strip()
    
    # Remove markdown formatting but keep the content
    response = MARKDOWN_PATTERN.sub('', response)
    
    # Remove common explanatory prefixes with less aggressive pattern matching
    original_length = len(response)
    for pattern in PREFIX_PATTERNS:
        response = pattern.sub('', response)
    
    # If cleaning removed too much content, revert to the original response
    if len(response) < original_length * 0.5 and original_length > 100:
        print("Cleaning removed too much content, reverting to original")
        response = response.strip()
    
    # Clean up multiple blank lines
    response = NEWLINE_PATTERN.sub('\n\n', response)
    
    # Return None if empty after cleaning
    if not response.strip():
        print("Response empty after cleaning")
        return None
    
    # Final validation - make sure we have a substantial response
    if len(response.strip()) < 100:
        print(f"Response too short after cleaning: {len(response.strip())} chars")
        return None
    
    return response.strip()

def detect_resume_sections(text):
    """Detect common resume sections for better context-aware prompting"""
    # Define patterns for common resume sections
    section_patterns = {
        "summary": re.compile(r'(professional\s+summary|summary\s+of\s+qualifications|profile|objective|about\s+me)', re.IGNORECASE),
        "experience": re.compile(r'(work\s+experience|professional\s+experience|employment|work\s+history)', re.IGNORECASE),
        "education": re.compile(r'(education|academic|qualifications|degrees)', re.IGNORECASE),
        "skills": re.compile(r'(skills|technical\s+skills|core\s+competencies|expertise)', re.IGNORECASE),
        "projects": re.compile(r'(projects|project\s+experience|portfolio)', re.IGNORECASE),
        "certifications": re.compile(r'(certifications|certificates|licenses)', re.IGNORECASE),
        "languages": re.compile(r'(languages|language\s+proficiency)', re.IGNORECASE),
        "awards": re.compile(r'(awards|honors|achievements)', re.IGNORECASE),
        "publications": re.compile(r'(publications|papers|research)', re.IGNORECASE),
        "volunteer": re.compile(r'(volunteer|community|service)', re.IGNORECASE)
    }
    
    # Find matches for each section
    detected_sections = []
    for section, pattern in section_patterns.items():
        if pattern.search(text):
            detected_sections.append(section)
    
    # If no sections detected but text is long enough, try to infer
    if not detected_sections and len(text) > 300:
        # Check for job titles with years (common in experience sections)
        if re.search(r'(19|20)\d{2}\s*(-|–|to)\s*(19|20)\d{2}|present', text, re.IGNORECASE):
            detected_sections.append("experience")
        
        # Check for education indicators
        if re.search(r'(bachelor|master|phd|degree|university|college|school)', text, re.IGNORECASE):
            detected_sections.append("education")
        
        # Check for technical terms (common in skills)
        if re.search(r'(proficient|familiar|knowledge|experience\s+with|skilled)', text, re.IGNORECASE):
            detected_sections.append("skills")
    
    return detected_sections

def ensure_paraphrase_results(resume_text, sections=None):
    """Ensure we get at least one paraphrased result, with multiple fallbacks"""
    # Try the simplest possible prompt first
    simple_prompt = f"Improve this resume text professionally:\n\n{resume_text}\n\nImproved resume:"
    
    try:
        response = query_llama(
            simple_prompt, 
            temperature=0.1,
            max_tokens=DEFAULT_MAX_TOKENS,
            model=FAST_MODEL,
            retries=2
        )
        result = clean_llama_response(response)
        if result and len(result) > len(resume_text) * 0.5:  # Ensure we got a meaningful response
            return {
                "type": "Enhanced",
                "text": result,
                "description": "Professional enhancement of your resume"
            }
    except Exception as e:
        print(f"Fallback error: {str(e)}")
    
    # Last resort - truncate and try again with a tiny prompt
    if len(resume_text) > 1000:
        truncated = resume_text[:1000] + "..."
        try:
            tiny_prompt = f"Rewrite this resume:\n\n{truncated}\n\nRewritten:"
            response = query_llama(tiny_prompt, 0.1, 500, FAST_MODEL, retries=2)
            result = clean_llama_response(response)
            if result and len(result) > 200:
                return {
                    "type": "Basic Enhancement",
                    "text": result + "\n\n[Note: This is a partial enhancement due to processing limitations.]",
                    "description": "Basic professional improvement of your resume"
                }
        except Exception:
            pass
    
    # If all else fails, return the original with a note
    return {
        "type": "Original",
        "text": resume_text,
        "description": "Original resume text (processing unavailable at this time)"
    }

if __name__ == "__main__":
    # Get server config from environment
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'false').lower() == 'true'  # Default to production mode
    
    # Use production WSGI server if available
    if os.getenv('USE_PRODUCTION_SERVER', 'true').lower() == 'true':
        from waitress import serve
        print(f"Starting production server on {host}:{port}")
        serve(app, host=host, port=port, threads=20)  # Increased threads
    else:
        app.run(debug=debug, host=host, port=port, threaded=True)