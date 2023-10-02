import azure.functions as func
import logging, os, json, openai , requests
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import Vector  

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="vector_search",methods=["POST"])
def vector_search(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # retrieve variables passed in via request body
    body = json.loads(req.get_body())
    #search_type = body["search_type"]
    search_query = body["search_query"]
    search_count = body["search_count"]
    index_name = body["index_name"]
    index_dimensions = body["index_dimensions"]
    index_vector_name = body["index_vector_name"]

    
    # retrieve environment variables  
    service_endpoint =os.getenv("ACS_ENDPOINT") 
    key = os.getenv("ACS_KEY") 
    
    # assemble azure credential object
    credential = AzureKeyCredential(key)


    # Pure Vector Search    
    search_client = SearchClient(service_endpoint, index_name, credential=credential)
    vector = Vector(value=generate_embeddings(index_dimensions,search_query), k=search_count, fields=index_vector_name)
    results = search_client.search(  
        search_text=None,  
        vectors= [vector],
        select=["title"],
    )  
    response = []

    for result in results: 
        response.append({
            "title" : result['title'],
            #"image_url": result["imageUrl"],
            "score": result['@search.score']
        })
        print(f"Title: {result['title']}")  
       # print(f"Score: {result['imageUrl']}")  
        print(f"Score: {result['@search.score']}")  
        
    response_body = { "results": response }
    response = func.HttpResponse(json.dumps(response_body))
    response.headers['Content-Type'] = 'application/json'    
    return response

# Function to generate embeddings for images
def generate_embeddings(dims,text):  

    if dims == 1024:
        cogSvcsEndpoint = os.environ["VISION_ENDPOINT"]  
        cogSvcsApiKey = os.environ["VISION_KEY"]  
        url = f"{cogSvcsEndpoint}/computervision/retrieval:vectorizeText"  
        params = {  
            "api-version": os.environ["VISION_API_VERSION"]  
        }  
        headers = {  
            "Content-Type": "application/json",  
            "Ocp-Apim-Subscription-Key": cogSvcsApiKey  
        }  
        data = {  
            "text": text  
        }  
        response = requests.post(url, params=params, headers=headers, json=data)  
        if response.status_code != 200:  
            print(f"Error: {response.status_code}, {response.text}")  
            response.raise_for_status()  
        embeddings = response.json()["vector"]  
    else:
        openai.api_type = os.getenv("OPENAI_API_TYPE")
        openai.api_key = os.getenv("OPENAI_API_KEY")  
        openai.api_base = os.getenv("OPENAI_API_ENDPOINT")  
        openai.api_version = os.getenv("OPENAI_API_VERSION")  
        response = openai.Embedding.create(
            input=text,
            engine="Text_Embed"
        )
        embeddings = response['data'][0]['embedding']

    return embeddings  