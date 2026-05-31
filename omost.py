import sys
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add the 'lib_omost' directory to sys.path
lib_omost_dir = os.path.join(current_dir, 'lib_omost')
if lib_omost_dir not in sys.path:
    sys.path.append(lib_omost_dir)

from lib_omost.canvas import Canvas as OmostCanvas, system_prompt

class OmostTool:
    def __init__(self, name, description, system_prompt):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.output_type = "canvas_conditioning"


    async def execute(self, args) -> Dict[str, Any]:

        prompt = args.get('input', '')
        llm_response = args.get('llm_response', '')
        

        
        try:
            canvas = OmostCanvas.from_bot_response(llm_response)
            canvas_conditioning = canvas.process()
            # Ensure canvas_conditioning is a flat list of dicts
            if (
                isinstance(canvas_conditioning, list)
                and len(canvas_conditioning) == 1
                and isinstance(canvas_conditioning[0], list)
            ):
                # Flatten once
                canvas_conditioning = canvas_conditioning[0]

            print("Canvas processed successfully")
            
            result = {
                self.output_type: canvas_conditioning,
                "prompt": prompt,
                "llm_response": llm_response
            }
        except Exception as e:
            logger.error(f"Error processing canvas: {str(e)}")
            result = {
                "error": str(e),
                "prompt": prompt,
                "llm_response": llm_response
            }
        

        return result

async def omost_function(args: Dict[str, Any]) -> Dict[str, Any]:
    tool = OmostTool(args['name'], args['description'], args['system_prompt'])
    result = await tool.execute(args)
    return result

