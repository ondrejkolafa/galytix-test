import click
import sys

from src.phrases_similarity import PhraseSimilarityCalculator
from src.utils import get_logger, read_yaml_file, write_json_file

logger = get_logger()

@click.command()
@click.option("--batch", "-b", is_flag=True, default=False, help='Use this to run the batch processing.')
@click.option("--model_path", "-m", default="", help='Use this to change the default Word2Vec model path.')
@click.option("--phrases_path", "-p", default="", help='Use this to choose different phrases file.')
def main(batch, model_path, phrases_path):
    config = read_yaml_file("config.yaml")
    if model_path:
        WORD2VEC_PATH = model_path
    else:    
        WORD2VEC_PATH = config['word2vec_path']
    if phrases_path:
        PHRASES_PATH = phrases_path
    else:
        PHRASES_PATH = config['phrases_path']
    OUTPUT_PATH = config['output_path']

    calculator = PhraseSimilarityCalculator(word2vec_path=WORD2VEC_PATH, phrases_path=PHRASES_PATH, distance_metric='cosine')

    calculator.batch_compute_embeddings()

    if batch:
        distance_matrix = calculator.batch_compute_distances()
        write_json_file(distance_matrix, OUTPUT_PATH)

    else:
        # Interactive querying mode loop
        while True:
            query_phrase = input("Enter a query and press Enter to find the closest match (or just hit enter to quit): ")  # Wait for user input
            if not query_phrase:
                logger.info("Exiting...")
                sys.exit(0)

            matches = calculator.find_closest_match(query_phrase, n=2)
            for phrase, distance in matches:
                logger.info(f"{query_phrase} matches: {phrase}, Distance: {distance:.4f}")


if __name__ == "__main__":
    main()