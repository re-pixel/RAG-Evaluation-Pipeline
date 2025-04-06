import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def intersection(range1, range2):
    if not range1 or not range2:
        return None
    start, end = max(range1[0], range2[0]), min(range1[1], range2[1])
    return (start, end) if start <= end else None

def union(ranges):
    ranges = sorted(ranges, key=lambda x: x[0])
    merged = [ranges[0]]
        
    for start, end in ranges[1:]:
        if start > merged[-1][1]:
            merged.append((start, end))
        elif end > merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        
    return merged

def difference(ranges, deduction):
    diff = []
    for interval in ranges:
        if intersection(interval, deduction) == None:
            diff.append(interval)
            continue
        if interval[0] < deduction[0]:
            diff.append((interval[0], deduction[0]))
        if interval[1] > deduction[1]:
            diff.append((deduction[1], interval[1]))
    
    return diff
        

def sum_of_ranges(ranges):
    ret = 0
    for start, end in ranges:
        ret += start-end
    return ret

def rigorous_document_search(document: str, target: str):
    """
    This function performs a rigorous search of a target string within a document. 
    It handles issues related to whitespace, changes in grammar, and other minor text alterations.
    The function first checks for an exact match of the target in the document. 
    If no exact match is found, it performs a raw search that accounts for variations in whitespace.
    If the raw search also fails, it splits the document into sentences and uses fuzzy matching 
    to find the sentence that best matches the target.
    
    Args:
        document (str): The document in which to search for the target.
        target (str): The string to search for within the document.

    Returns:
        tuple: A tuple containing the best match found in the document, its start index, and its end index.
        If no match is found, returns None.
    """
    if target.endswith('.'):
        target = target[:-1]
    
    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index
    else:
        raw_search = find_query_despite_whitespace(document, target)
        if raw_search is not None:
            return raw_search

    # Split the text into sentences
    sentences = re.split(r'[.!?]\s*|\n', document)

    # Find the sentence that matches the query best
    best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)

    if best_match[1] < 98:
        return None
    
    reference = best_match[0]

    start_index = document.find(reference)
    end_index = start_index + len(reference)

    return reference, start_index, end_index

def find_query_despite_whitespace(document, query):

    # Normalize spaces and newlines in the query
    normalized_query = re.sub(r'\s+', ' ', query).strip()
    
    # Create a regex pattern from the normalized query to match any whitespace characters between words
    pattern = r'\s*'.join(re.escape(word) for word in normalized_query.split())
    
    # Compile the regex to ignore case and search for it in the document
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)
    
    if match:
        return document[match.start(): match.end()], match.start(), match.end()
    else:
        return None