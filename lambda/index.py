# lambda/index.py
import json
import urllib.request

def lambda_handler(event, context):
    try:
        url = "https://a475-14-10-136-97.ngrok-free.app" # ハードコーディングよくない？
        request_body = json.loads(event['body'])
        request_body['invoked_function_arn'] = context.invoked_function_arn
        
        req = urllib.request.Request(
            url,
            data=json.dumps(request_body).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(req) as response:
            res_body = response.read()
            return json.loads(res_body)
        
    except Exception as error:
        print("Error:", str(error))
        return {
            "statusCode": 500,
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }