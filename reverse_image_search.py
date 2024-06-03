import gradio as gr
from models import Resnet50
from vectordb import MilvusDB
from PIL import Image

COLLECTION_NAME = 'liam_images2'
DIM = 2048

model = Resnet50()
db = MilvusDB()

# 构建图片集
def index_image(indexing_data_path):
    print(f"Indexing with data from: {indexing_data_path}")
    # 获取指定路径下的所有图片，并抽取特征
    image_paths, features = model.batch_extract_features_by_parent_path(indexing_data_path)
    # 创建数据库表
    db.create_milvus_collection(COLLECTION_NAME, DIM)
    user_data = [image_paths, features]
    # 存入特征
    nums = db.insert_data(user_data)
    return f"Indexed {nums} images."

# 查找相似
def search_similar_images(image_path):
    # 抽取样本图片特征
    key_feature = model.extract_feature(image_path)
    # 连接数据库表
    db.connect_collection(COLLECTION_NAME)
    # 查找相似
    similar_images_paths = db.search_data(key_feature.reshape(1, -1))
    return [Image.open(path) for path in similar_images_paths]

def process_image(image_file):
    image = Image.open(image_file)
    image = image.convert('RGB')
    return image

# 使用 Gradio Blocks 创建 UI
with gr.Blocks() as demo:
    image_data_path = gr.Textbox(label="Enter the image directory path", lines=1, placeholder="/path/to/indexing/data")
    index_button = gr.Button("Indexing images")
    index_output = gr.Textbox(label="Indexing Output")    
    image_path = gr.Image(label="Upload an image", type="filepath")
    # 创建按钮以触发搜索相似图片
    search_button = gr.Button("Search Similar Images")
    search_output = gr.Gallery(label="Search Results")

    # 将按钮与函数关联
    index_button.click(index_image, inputs=image_data_path, outputs=index_output)
    search_button.click(search_similar_images, inputs=image_path, outputs=search_output)

# 运行界面
demo.launch()
