# lambda/index.py
import json
import urllib.request

def lambda_handler(event, context):
    try:
        url = "https://970c-133-3-201-38.ngrok-free.app" # ハードコーディングよくない？
        request_body = json.loads(event['body'])
        request_body['invoked_function_arn'] = context.invoked_function_arn
        
        req = urllib.request.Request(
            url,
            data=json.dumps(request_body).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(req) as response:
            res_body = response.read()
            res_json = json.loads(res_body)
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps({
                    "success": True,
                    "response": res_json["generated_text"],
                    "conversationHistory": res_json["conversation_history"]
                })
            }
        
    except Exception as error:
        print("Error:", str(error))
        return {
            "statusCode": 500,
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }