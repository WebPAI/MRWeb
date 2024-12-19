from flask import Flask, render_template, request, jsonify
import sys
sys.path.append('../../')
from bots import GPT4
import re

prompt_direct = """Here is a screenshot of a web page and its "action list" which specifies the links and images in the webpage. Please write a HTML, Tailwind CSS, and javascript to make it look exactly like the original web page. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. The format of the action list is as follows:
    {
    "position": bounding box of format [[x1, y1], [x2, y2]], specifying the top left corner and the bottom right corner of the element;
    "type": element type;
    "on_click_jump_to": url of the element (the name is "src" for images);
    }
The action list is as follows: 

[ACTION_LIST]

"""

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('tool.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    image_base64 = data.get('image')
    action_list = data.get('actionList', [])
    model = data.get('model')
    key = data.get('key')
    scale = data.get('scale', 1)

    for action in action_list:
        value = action["position"][0]
        action["position"] = [int(value[0] / scale), int(value[1] / scale), int(value[2] / scale), int(value[3] / scale)]
        print(action["position"])

    bot = GPT4(key)
    try:
        raw = bot.ask(prompt_direct.replace("[ACTION_LIST]", str(action_list)), image_base64)
        # find the html code in the response ```html ... ```
        response = re.findall(r"```html([^`]+)```", raw)
        if not response:
            response = raw
        
    except Exception as e:
        response = str(e)

    return jsonify({"response": response})



if __name__ == '__main__':
    app.run(debug=True)
