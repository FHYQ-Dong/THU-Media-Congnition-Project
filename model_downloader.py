import requests, os
from tqdm import tqdm

def download(model_name: str, path: str = r'jameslahm', repo: str = "github.com/THU-MIG/yolov10/releases/download/v1.1"):
    pretrained_models = [
        'yolov10b',
        'yolov10l',
        'yolov10m',
        'yolov10n',
        'yolov10s',
        'yolov10x'
    ]
    assert model_name in pretrained_models
    
    model_name = f"{model_name}.pt"
    model_path = os.path.join(path, model_name)
    if os.path.exists(model_path):
        return
    
    if not os.path.exists(path):
        os.makedirs(path)    
    for i in range(3):
        try:
            url = f"https://{repo}/{model_name}"
            resp = requests.get(url, stream=True)
            total_size = int(resp.headers.get('content-length', 0))
            block_size = 1024
            t = tqdm(
                total = total_size, 
                unit = 'B', 
                unit_scale =True,
                desc = f"Downloading {model_name}",
                ascii=True,
                ncols=100
            )
            
            with open(model_path, 'wb') as f:
                for data in resp.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()
            break
        except Exception as e:
            print(f"Download failed with error: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
    if not os.path.exists(model_path):
        print(f"Download {model_name} failed.")
    else:
        print(f"Model {model_name} saved to {model_path}.")
