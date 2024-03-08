import numpy as np


def compute_prototypes(m_embeddings, m_labels, m_logger):
    unique_labels = np.unique(m_labels)
    m_prototypes = {}
    for label in unique_labels:
        indices = np.where(m_labels == label)[0]
        class_embeddings = m_embeddings[indices]
        m_prototypes[label] = np.mean(class_embeddings, axis=0)
        m_logger.log_embedding(m_prototypes[label], label, class_embeddings.shape[0])
    return m_prototypes


def compute_own_class_prototype_probabilities(m_prototypes, m_embeddings, m_labels):
    own_class_prototype_probs = []
    for i in range(len(m_embeddings)):
        embedding = m_embeddings[i]
        label = m_labels[i]
        true_prototype = m_prototypes[str(label[0])]
        true_dst = np.linalg.norm(embedding - true_prototype)
        false_dst = np.inf
        for cls_label in m_prototypes.keys():
            if cls_label == str(label[0]):
                continue
            query_prototype = m_prototypes[cls_label]
            false_dst = min([np.linalg.norm(embedding - query_prototype), false_dst])
        own_class_prototype_prob = false_dst / (true_dst + false_dst)
        own_class_prototype_probs.append(own_class_prototype_prob)
    return own_class_prototype_probs
