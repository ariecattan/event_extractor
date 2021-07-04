import argparse
import pyhocon
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from models import SpanScorer, SpanEmbedder
from utils import *
from model_utils import *
import json




def init_models(config, device):
    span_repr = SpanEmbedder(config, device).to(device)
    span_repr.load_state_dict(torch.load(os.path.join(config['model_path'], "events_span_repr"),
                                         map_location=device))
    span_repr.eval()
    span_scorer = SpanScorer(config).to(device)
    span_scorer.load_state_dict(torch.load(os.path.join(config['model_path'], "events_span_scorer"),
                                           map_location=device))

    return span_repr, span_scorer



def is_included(docs, starts, ends, i1, i2):
    doc1, start1, end1 = docs[i1], starts[i1], ends[i1]
    doc2, start2, end2 = docs[i2], starts[i2], ends[i2]

    if doc1 == doc2 and (start1 >= start2 and end1 <= end2):
        return True
    return False


def remove_nested_mentions(cluster_ids, doc_ids, starts, ends):
    # nested_mentions = collections.defaultdict(list)
    # for i, x in range(len(cluster_ids)):
    #     nested_mentions[x].append(i)

    doc_ids = np.asarray(doc_ids)
    starts = np.asarray(starts)
    ends = np.asarray(ends)

    new_cluster_ids, new_docs_ids, new_starts, new_ends = [], [], [], []

    for cluster, idx in cluster_ids.items():
        docs = doc_ids[idx]
        start = starts[idx]
        end = ends[idx]


        for i in range(len(idx)):
            indicator = [is_included(docs, start, end, i, j) for j in range(len(idx))]
            if sum(indicator) > 1:
                continue

            new_cluster_ids.append(cluster)
            new_docs_ids.append(docs[i])
            new_starts.append(start[i])
            new_ends.append(end[i])


    clusters = collections.defaultdict(list)
    for i, cluster_id in enumerate(new_cluster_ids):
        clusters[cluster_id].append(i)

    return clusters, new_docs_ids, new_starts, new_ends





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file(args.config)
    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    create_folder(config['save_path'])
    device = 'cuda:{}'.format(config['gpu_num'][0]) if torch.cuda.is_available() else 'cpu'


    # Load models and init clustering
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    config['bert_hidden_size'] = bert_model.config.hidden_size
    span_repr, span_scorer = init_models(config, device)


    # Load data
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    data = create_corpus(config, bert_tokenizer)

    doc_ids, sentence_ids, starts, ends = [], [], [], []
    all_topic_predicted_clusters = []
    max_cluster_id = 0

    # Go through each topic

    for topic_num, topic in enumerate(data.topic_list):
        print('Processing topic {}'.format(topic))
        docs_embeddings, docs_length = pad_and_read_bert(data.topics_bert_tokens[topic_num], bert_model)
        span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
            config, data, topic_num, docs_embeddings, docs_length)

        doc_id, sentence_id, start, end = span_meta_data
        start_end_embeddings, continuous_embeddings, width = span_embeddings
        width = width.to(device)


        all_span_scores = []
        for i in tqdm(range(0, len(width), 4)):
            with torch.no_grad():
                span_emb = span_repr(start_end_embeddings[i:i+4], continuous_embeddings[i:i+4], width[i:i+4])
                span_scores = span_scorer(span_emb)
            all_span_scores.extend(span_scores)
        all_span_scores = torch.tensor(all_span_scores)

        if config.exact:
            span_indices = torch.where(all_span_scores > 0)[0]
        else:
            k = int(config['top_k'] * num_of_tokens)
            _, span_indices = torch.topk(span_scores.squeeze(1), k, sorted=False)


        print(f"span indices shape {span_indices.shape}")
        number_of_mentions = len(span_indices)
        start_end_embeddings = start_end_embeddings[span_indices]
        continuous_embeddings = [continuous_embeddings[i] for i in span_indices]
        width = width[span_indices]
        torch.cuda.empty_cache()


        doc_ids.extend(doc_id[span_indices.cpu()])
        sentence_ids.extend(sentence_id[span_indices].tolist())
        starts.extend(start[span_indices].tolist())
        ends.extend(end[span_indices].tolist())
        torch.cuda.empty_cache()


    t2s = collections.defaultdict(dict)
    for doc, tokens in data.documents.items():
        for sent_id, token_id, _, _ in tokens:
            t2s[doc][token_id] = sent_id

    mentions = [{
        "topic": "0",
        "subtopic": "0_0",
        "doc_id": doc,
        "sentence_id": t2s[doc][start],
        "tokens_ids": list(range(start, end +1 )),
        "tokens": " ".join([x[2] for x in data.documents[doc][start-1:end]]),
        "cluster_id": 0,
        "m_id": i
    } for i, (doc, start, end) in enumerate(zip(doc_ids, starts, ends))]


    with open(config.output_path, 'w') as f:
        json.dump(mentions, f, indent=4)