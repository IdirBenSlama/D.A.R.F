import os
import requests
import zipfile
import pandas as pd
import json
import logging
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create data directories if they don't exist
datasets_dir = Path("datasets")
downloads_dir = datasets_dir / "downloads"
extracted_dir = datasets_dir / "extracted"
processed_dir = datasets_dir / "processed"

for directory in [datasets_dir, downloads_dir, extracted_dir, processed_dir]:
    directory.mkdir(exist_ok=True)

def download_file(url, target_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as f, tqdm(
            desc=target_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    
    return target_path

def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info(f"Extracted {zip_path} to {extract_to}")
    return extract_to

def process_iris_dataset():
    """Process the Iris dataset for RZSet vectors"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    target_path = downloads_dir / "iris.data"
    
    # Download if not exists
    if not target_path.exists():
        logger.info(f"Downloading Iris dataset from {url}")
        download_file(url, target_path)
    
    # Process into vectors format
    logger.info("Processing Iris dataset into vector format")
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df = pd.read_csv(target_path, header=None, names=column_names)
    
    # Convert to vectors
    vectors = []
    for idx, row in df.iterrows():
        vector = {
            "id": f"iris-{idx}",
            "vector": [float(row["sepal_length"]), float(row["sepal_width"]), 
                     float(row["petal_length"]), float(row["petal_width"])],
            "metadata": {
                "class": row["class"],
                "source": "iris",
                "description": "Iris flower dataset"
            }
        }
        vectors.append(vector)
    
    # Save processed data
    output_path = processed_dir / "iris_vectors.json"
    with open(output_path, 'w') as f:
        json.dump({"vectors": vectors}, f)
    
    logger.info(f"Saved processed Iris vectors to {output_path}")
    return output_path

def process_wine_dataset():
    """Process the Wine dataset for knowledge graph"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    target_path = downloads_dir / "wine.data"
    
    # Download if not exists
    if not target_path.exists():
        logger.info(f"Downloading Wine dataset from {url}")
        download_file(url, target_path)
    
    # Process into knowledge graph format
    logger.info("Processing Wine dataset into knowledge graph format")
    column_names = ["class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
                    "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
                    "proanthocyanins", "color_intensity", "hue", "od280_od315", "proline"]
    df = pd.read_csv(target_path, header=None, names=column_names)
    
    # Convert to graph structure
    nodes = []
    edges = []
    
    # Create class nodes
    for class_id in df["class"].unique():
        nodes.append({
            "id": f"class-{class_id}",
            "label": f"Wine Class {class_id}",
            "type": "Class",
            "size": 10
        })
    
    # Create wine sample nodes and edges
    for idx, row in df.iterrows():
        # Create sample node
        sample_id = f"wine-{idx}"
        nodes.append({
            "id": sample_id,
            "label": f"Wine Sample {idx}",
            "type": "Sample",
            "size": 5,
            "properties": {
                "alcohol": float(row["alcohol"]),
                "malic_acid": float(row["malic_acid"]),
                "color_intensity": float(row["color_intensity"])
            }
        })
        
        # Connect to class
        edges.append({
            "id": f"edge-class-{idx}",
            "source": sample_id,
            "target": f"class-{row['class']}",
            "label": "BELONGS_TO"
        })
        
        # Create attribute nodes for significant properties
        for attr in ["alcohol", "malic_acid", "magnesium", "color_intensity"]:
            # Only create nodes for values above average
            if row[attr] > df[attr].mean():
                attr_id = f"{attr}-{idx}"
                nodes.append({
                    "id": attr_id,
                    "label": f"{attr.replace('_', ' ').title()}: {row[attr]:.2f}",
                    "type": "Attribute",
                    "size": 3
                })
                
                edges.append({
                    "id": f"edge-attr-{idx}-{attr}",
                    "source": sample_id,
                    "target": attr_id,
                    "label": "HAS_PROPERTY"
                })
    
    # Save processed data
    output_path = processed_dir / "wine_knowledge_graph.json"
    with open(output_path, 'w') as f:
        json.dump({"nodes": nodes, "edges": edges}, f)
    
    logger.info(f"Saved processed Wine knowledge graph to {output_path}")
    return output_path

def process_text_dataset():
    """Process the BBC News dataset for LLM data"""
    url = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
    target_path = downloads_dir / "bbc-fulltext.zip"
    extract_path = extracted_dir / "bbc"
    
    # Download if not exists
    if not target_path.exists():
        logger.info(f"Downloading BBC News dataset from {url}")
        download_file(url, target_path)
    
    # Extract if not already extracted
    if not extract_path.exists():
        logger.info(f"Extracting {target_path}")
        extract_zip(target_path, extracted_dir)
    
    # Process into LLM data
    logger.info("Processing BBC News dataset into LLM context data")
    
    documents = []
    
    # Iterate through directories (categories)
    categories = [d for d in extract_path.iterdir() if d.is_dir()]
    for category_dir in categories:
        category = category_dir.name
        
        # Process each article in the category
        for article_file in category_dir.glob("*.txt"):
            with open(article_file, 'r', encoding='latin1') as f:
                content = f.read().strip()
                
                # Extract title and body
                lines = content.split('\n')
                title = lines[0] if lines else "Untitled"
                body = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                
                documents.append({
                    "id": article_file.stem,
                    "title": title,
                    "category": category,
                    "content": body,
                    "source": "BBC News Dataset",
                    "metadata": {
                        "file": str(article_file.relative_to(extracted_dir)),
                        "length": len(body),
                        "word_count": len(body.split())
                    }
                })
    
    # Save processed data
    output_path = processed_dir / "bbc_news_documents.json"
    with open(output_path, 'w') as f:
        json.dump({"documents": documents}, f)
    
    logger.info(f"Saved processed BBC News documents to {output_path}")
    return output_path

def main():
    """Download and process all datasets"""
    logger.info("Starting dataset download and processing")
    
    # Process all datasets
    iris_path = process_iris_dataset()
    wine_path = process_wine_dataset()
    bbc_path = process_text_dataset()
    
    # Create a dataset info file
    dataset_info = {
        "datasets": [
            {
                "name": "Iris Dataset",
                "source": "UCI Machine Learning Repository",
                "description": "Famous dataset for classification containing 3 classes of 50 instances each, where each class refers to a type of iris plant.",
                "processed_file": str(iris_path),
                "record_count": 150,
                "features": 4,
                "usage": "RZSet Vector Store"
            },
            {
                "name": "Wine Dataset",
                "source": "UCI Machine Learning Repository",
                "description": "Results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.",
                "processed_file": str(wine_path),
                "record_count": 178,
                "features": 13,
                "usage": "Knowledge Graph"
            },
            {
                "name": "BBC News Dataset",
                "source": "University College Dublin",
                "description": "Collection of news articles from the BBC for use in text classification.",
                "processed_file": str(bbc_path),
                "record_count": 2225,
                "categories": ["business", "entertainment", "politics", "sport", "tech"],
                "usage": "LLM Knowledge Base"
            }
        ],
        "processed_time": pd.Timestamp.now().isoformat()
    }
    
    # Save dataset info
    info_path = datasets_dir / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"Dataset processing complete. Info saved to {info_path}")
    logger.info(f"Total datasets processed: 3")
    
    # Print summary
    print("\nDatasets processed successfully:")
    print(f"1. Iris Dataset (150 records) - For RZSet Vector Store")
    print(f"2. Wine Dataset (178 records) - For Knowledge Graph")
    print(f"3. BBC News Dataset (2225 documents) - For LLM Knowledge Base")
    print(f"\nProcessed data stored in: {processed_dir}")

if __name__ == "__main__":
    main()
