import json
import re
import csv
from typing import Dict, Any, List, Tuple
from io import StringIO
import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


class ContestantAgent:
    def __init__(self, api_key: str):
        """
        Initialize the agent with the provided API key.
        Uses pydantic-ai framework for robust agent loop handling.
        """
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        
        self.api_key = api_key
        
        # CRITICAL: Must set http_client manually for judging system compatibility
        provider = OpenAIProvider(
            api_key=api_key,
            base_url="https://api.metisai.ir/openai/v1",
            http_client=httpx.AsyncClient()  # Don't forget!
        )
        
        model = OpenAIChatModel(
            provider=provider,
            model_name="gpt-4.1-mini"
        )
        
        # Create pydantic-ai agent with system prompt
        self.agent = Agent(
            model=model,
            system_prompt="""You are an expert problem-solving agent.

CRITICAL RULES:
1. Return ONLY the final answer - no explanations, prefixes, or extra text
2. For numbers: return just the number (e.g., "42" not "The answer is 42")
3. For text: return just the text without quotes
4. Use tools when you need external data (URLs, CSV files, APIs)
5. For simple questions, answer directly without tools
6. Learn from previous attempt history to avoid repeated mistakes

STRATEGY:
- Simple math/knowledge → answer directly
- URLs mentioned → use http_request or CSV tools
- CSV files → use get_csv_row_count, get_csv_summary, or find_csv_row
- Persian math → use calculate
- API endpoints → use http_request""",
        )
        
        # Register all tools with the agent
        self._register_tools()
    
    def _register_tools(self):
        """Register all tools with the pydantic-ai agent."""
        
        # Tool 1: HTTP Request (replaces read_url_content, click_website, api_request)
        @self.agent.tool_plain
        def http_request(url: str, method: str = "GET", params: dict = None, body: dict = None) -> str:
            """
            Make an HTTP request to a URL. Use for reading web content, APIs, or text files.
            
            Args:
                url: The URL to request
                method: HTTP method (GET, POST, PUT, DELETE). Default is GET.
                params: Optional query parameters
                body: Optional request body for POST/PUT
            """
            try:
                if method == "GET":
                    response = httpx.get(url, params=params, timeout=30.0, follow_redirects=True)
                elif method == "POST":
                    response = httpx.post(url, params=params, json=body, timeout=30.0, follow_redirects=True)
                elif method == "PUT":
                    response = httpx.put(url, params=params, json=body, timeout=30.0, follow_redirects=True)
                elif method == "DELETE":
                    response = httpx.delete(url, params=params, timeout=30.0, follow_redirects=True)
                else:
                    return f"Unsupported HTTP method: {method}"
                
                response.raise_for_status()
                return response.text
            except Exception as e:
                return f"Error making HTTP request: {str(e)}"
        
        # Tool 2: Get CSV Row Count (simplified from analyze_csv)
        @self.agent.tool_plain
        def get_csv_row_count(url: str) -> str:
            """
            Count the number of rows in a CSV file from a URL.
            
            Args:
                url: The URL of the CSV file
            
            Returns:
                The number of data rows (excluding header) as a string
            """
            try:
                response = httpx.get(url, timeout=30.0, follow_redirects=True)
                response.raise_for_status()
                
                csv_reader = csv.DictReader(StringIO(response.text))
                rows = list(csv_reader)
                
                return str(len(rows))
            except Exception as e:
                return f"Error counting CSV rows: {str(e)}"
        
        # Tool 3: Get CSV Summary (simplified from analyze_csv)
        @self.agent.tool_plain
        def get_csv_summary(url: str) -> str:
            """
            Get a summary of a CSV file including headers, row count, and sample data.
            CRITICAL: Returns only a small summary to avoid token limits.
            
            Args:
                url: The URL of the CSV file
            
            Returns:
                JSON string with row_count, headers, and 3 sample rows
            """
            try:
                response = httpx.get(url, timeout=30.0, follow_redirects=True)
                response.raise_for_status()
                
                csv_reader = csv.DictReader(StringIO(response.text))
                rows = list(csv_reader)
                
                summary = {
                    "row_count": len(rows),
                    "headers": list(rows[0].keys()) if rows else [],
                    "sample_rows": rows[:3] if len(rows) > 3 else rows  # Only 3 samples to save tokens
                }
                
                return json.dumps(summary, ensure_ascii=False)
            except Exception as e:
                return f"Error getting CSV summary: {str(e)}"
        
        # Tool 4: Find CSV Row by ID (simplified from analyze_csv)
        @self.agent.tool_plain
        def find_csv_row(url: str, id_field: str, id_value: str) -> str:
            """
            Find a specific row in a CSV file by ID field and value.
            
            Args:
                url: The URL of the CSV file
                id_field: The name of the ID column (e.g., "employee_id", "id")
                id_value: The value to search for
            
            Returns:
                JSON string of the matching row, or error message if not found
            """
            try:
                response = httpx.get(url, timeout=30.0, follow_redirects=True)
                response.raise_for_status()
                
                csv_reader = csv.DictReader(StringIO(response.text))
                rows = list(csv_reader)
                
                for row in rows:
                    if str(row.get(id_field, "")).strip() == str(id_value).strip():
                        return json.dumps(row, ensure_ascii=False)
                
                return f"No row found with {id_field}={id_value}"
            except Exception as e:
                return f"Error finding CSV row: {str(e)}"
        
        # Tool 5: Count Entities (Improved with whole-word matching)
        @self.agent.tool_plain
        def count_entities(urls: List[str], entity_field: str) -> str:
            """
            Count specific entities across one or more URLs.
            For JSON, counts items in lists.
            For text, counts whole-word occurrences.
            
            Args:
                urls: List of URLs to check
                entity_field: The field name or word to count
            
            Returns:
                Total count as a string
            """
            try:
                total_count = 0
                for url in urls:
                    response = httpx.get(url, timeout=30.0, follow_redirects=True)
                    response.raise_for_status()
                    content = response.text
                    
                    # Try JSON parsing first
                    try:
                        data = json.loads(content)
                        if isinstance(data, list):
                            # Prompt is likely "count items"
                            total_count += len(data)
                        elif isinstance(data, dict):
                            # Prompt is "count entity_field"
                            if entity_field in data and isinstance(data[entity_field], list):
                                total_count += len(data[entity_field])
                            elif entity_field in data:
                                total_count += 1  # Count the key itself
                    
                    except json.JSONDecodeError:
                        # Fallback to whole-word text counting
                        # \b ensures we match "apple" but not "apples"
                        matches = re.findall(r'\b' + re.escape(entity_field.lower()) + r'\b', content.lower())
                        total_count += len(matches)
                
                return str(total_count)
            except Exception as e:
                return f"Error counting entities: {str(e)}"
        
        # Tool 6: Calculate (with Persian support)
        @self.agent.tool_plain
        def calculate(expression: str) -> str:
            """
            Perform mathematical calculations. Supports Persian/Farsi text.
            Examples: "دو به علاوه دو" → "4", "10 + 15" → "25"
            
            Args:
                expression: The mathematical expression (Persian or numeric)
            
            Returns:
                The calculated result as a string
            """
            try:
                # Persian number mapping
                persian_numbers = {
                    'صفر': '0', 'یک': '1', 'دو': '2', 'سه': '3', 'چهار': '4',
                    'پنج': '5', 'شش': '6', 'هفت': '7', 'هشت': '8', 'نه': '9',
                    'ده': '10', 'یازده': '11', 'دوازده': '12', 'سیزده': '13',
                    'چهارده': '14', 'پانزده': '15', 'شانزده': '16', 'هفده': '17',
                    'هجده': '18', 'نوزده': '19', 'بیست': '20',
                    'سی': '30', 'چهل': '40', 'پنجاه': '50', 'شصت': '60',
                    'هفتاد': '70', 'هشتاد': '80', 'نود': '90', 'صد': '100'
                }
                
                # Persian operator mapping
                persian_operators = {
                    'به علاوه': ' + ', 'بعلاوه': ' + ', 'علاوه': ' + ', 'جمع': ' + ',
                    'به اضافه': ' + ', 'اضافه': ' + ',
                    'منهای': ' - ', 'منها': ' - ', 'تفریق': ' - ', 'کم': ' - ',
                    'ضربدر': ' * ', 'ضرب در': ' * ', 'در': ' * ', 'ضرب': ' * ',
                    'تقسیم بر': ' / ', 'تقسیم': ' / ', 'بر': ' / '
                }
                
                # Convert Persian to math expression
                expr = expression.lower().strip()
                
                # Replace numbers (longest first to avoid partial matches)
                for persian in sorted(persian_numbers.keys(), key=len, reverse=True):
                    expr = expr.replace(persian, persian_numbers[persian])
                
                # Replace operators (longest first)
                for persian in sorted(persian_operators.keys(), key=len, reverse=True):
                    expr = expr.replace(persian, persian_operators[persian])
                
                # Clean and evaluate
                expr = re.sub(r'[^\d+\-*/().\s]', '', expr)
                expr = ' '.join(expr.split())
                
                result = eval(expr)
                
                # Return integer if whole number
                if isinstance(result, float) and result.is_integer():
                    return str(int(result))
                
                return str(result)
            except Exception as e:
                return f"Error calculating: {str(e)}"
    
    def solve_lock(self, problem: Dict[str, Any], history: List[Tuple[str, str]]) -> str:
        """
        Solve a single lock challenge using pydantic-ai agent framework.
        
        Args:
            problem: Dictionary containing the problem information
            history: List of tuples (previous_answer, feedback) from failed attempts
        
        Returns:
            str: The final answer (clean, no explanations)
        """
        try:
            # Build comprehensive problem text from ALL fields in the problem dictionary
            problem_text_parts = []
            if isinstance(problem, dict):
                # Iterate over all key-value pairs to capture URLs, files, etc.
                for key, value in problem.items():
                    if isinstance(value, str):
                        # Format: "key: value"
                        problem_text_parts.append(f"{key}: {value}")
                    elif value is not None:
                        # Handle non-string values (numbers, booleans, etc.)
                        problem_text_parts.append(f"{key}: {value}")
                
                # Join all parts into comprehensive prompt
                problem_text = "\n".join(problem_text_parts)
                
                # Fallback if dict has no valid fields
                if not problem_text:
                    problem_text = str(problem)
            else:
                # Handle non-dict problems
                problem_text = str(problem)
            
            # Build context from history
            if history:
                history_context = "--- PREVIOUS FAILED ATTEMPTS ---\n"
                for i, (prev_answer, feedback) in enumerate(history, 1):
                    history_context += f"Attempt {i}:\n"
                    history_context += f"  My Answer: \"{prev_answer}\"\n"
                    history_context += f"  Feedback: \"{feedback}\"\n"
                history_context += "--- END OF HISTORY ---\n\n"
                
                # Prepend history to problem
                problem_text = f"{history_context}--- CURRENT PROBLEM ---\n{problem_text}"
            
            # Use pydantic-ai to handle the entire agent loop
            result = self.agent.run_sync(problem_text)
            
            # Extract and clean the answer
            answer = str(result.output).strip()
            answer = self._clean_answer(answer)
            
            return answer
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _clean_answer(self, answer: str) -> str:
        """
        Aggressively clean the answer to find the final result.
        It prioritizes finding the *last number* in the output.
        If no number is found, it falls back to taking the last non-empty line for text.
        """
        if not answer:
            return ""

        # 1. Try to find the last number (integer or float)
        # This regex finds all numbers, e.g., "123", "45.67", "-8"
        numbers = re.findall(r"[-+]?\d*\.?\d+", answer)
        
        if numbers:
            # Return the very last number found in the agent's output
            return numbers[-1]
        
        # 2. If NO numbers were found, fall back to last non-empty line logic
        #    (for text-based answers like Lock 3: "Analyze Text File")
        lines = answer.strip().split('\n')
        last_line = ""
        for line in reversed(lines):
            stripped_line = line.strip()
            if stripped_line:
                last_line = stripped_line
                break
        
        if not last_line:
            return ""  # Should be rare, but safe

        # 3. Run prefix stripping on the last line
        prefixes = [
            "the answer is ", "answer: ", "final answer: ",
            "result: ", "output: ", "solution: ", "response: "
        ]
        
        last_line_lower = last_line.lower()
        for prefix in prefixes:
            if last_line_lower.startswith(prefix):
                last_line = last_line[len(prefix):].strip()
                break
        
        # 4. Run quote stripping
        if (last_line.startswith('"') and last_line.endswith('"')) or \
           (last_line.startswith("'") and last_line.endswith("'")):
            last_line = last_line[1:-1].strip()
        
        return last_line
    
    def choose_path(self, scenario_prompt: str) -> List[str]:
        """
        Choose a path for Phase 2 competition.
        
        Args:
            scenario_prompt: Description of the scenario and available paths
        
        Returns:
            List[str]: The chosen path as a list of steps/decisions
        """
        try:
            result = self.agent.run_sync(scenario_prompt)
            answer = str(result.output).strip()
            
            # Try to parse as JSON list
            try:
                paths = json.loads(answer)
                if isinstance(paths, list):
                    return paths
            except:
                pass
            
            # Otherwise split by common delimiters
            paths = [p.strip() for p in answer.split(',')]
            return paths
            
        except Exception as e:
            return [f"Error: {str(e)}"]
