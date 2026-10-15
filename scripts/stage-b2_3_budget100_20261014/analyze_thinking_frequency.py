"""Read-only frequency analysis of externally generated, visible Qwen traces.

Use one 32K-cap trace per input as the primary corpus, rather than pooling
the three mostly identical budget prefixes as independent observations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/stage_b2_3_budget100_20261014"
OUT = SOURCE / "analysis_high_frequency_20261015"
CAPS = (8192,16384,32768)
WORDS = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?")
PHRASES = {
    "maybe": (r"\bmaybe\b", "猜测／不确定表达"),
    "perhaps": (r"\bperhaps\b", "猜测／不确定表达"),
    "wait": (r"\bwait\b", "中断、回看或自我检查"),
    "let me think": (r"\blet me think\b", "思考过程提示语"),
    "let's think": (r"\blet's think\b", "思考过程提示语"),
    "let me try": (r"\blet me try\b", "尝试新推导或布局"),
    "alternatively": (r"\balternatively\b", "切换假设／推导方式"),
    "I'm not sure": (r"\bi'm not sure\b", "不确定表达"),
    "not necessarily": (r"\bnot necessarily\b", "否定必然性"),
    "for example": (r"\bfor example\b", "构造具体例子"),
    "guess / guessing": (r"\bguess(?:ing|ed|es)?\b", "猜测表达"),
    "ambiguous / ambiguity": (r"\bambig(?:uous|uity)\b", "歧义表达"),
    "not enough information": (r"\bnot enough information\b", "信息不足表达"),
    "cannot determine": (r"\bcannot determine\b", "无法确定表达"),
    "coordinates": (r"\bcoordinates\b", "坐标构造"),
    "coordinate system": (r"\bcoordinate system\b", "坐标构造"),
    "unit steps": (r"\bunit steps\b", "单位步长约定"),
    "clock": (r"\bclock\b", "时钟方位理解"),
    "the question is": (r"\bthe question is\b", "重新陈述任务"),
    "but how": (r"\bbut how\b", "追问或推导停滞"),
    "to the left of": (r"\bto the left of\b", "空间关系重述"),
    "to the right of": (r"\bto the right of\b", "空间关系重述"),
    "same level": (r"\bsame level\b", "对齐／抵消关系"),
    "Final answer": (r"\bfinal answer\b", "思考区中的答案字样，非正式最终输出"),
}
STOP = set("a an the i me my we us our you your he him his she her it its they them their this that these those is am are was were be been being do does did have has had can could would should will shall may might must to of in on at by for from with as and or but if then than so because while since when where what which who how not no yes all some any each both either neither more most other another one two there here out up down into over under again also just very really now let let's s t m re ve ll d".split())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def normalize(text):
    return text.replace("\u2019", "'").replace("\u2018", "'").casefold()


def sentence_parts(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    for part in parts:
        part = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", part).strip()
        key = re.sub(r"\s+", " ", normalize(part)).strip(" \t.,;:!?")
        words = WORDS.findall(key)
        if words:
            yield key, part, len(words)


def writecsv(name, rows):
    with (OUT/name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    audit = read(SOURCE/"audit_completion.json")
    assert audit["status"] == "PASS"
    corpora, row_metrics, provenance = {}, [], []
    primary_sentence_counts, primary_sentence_docs = Counter(), defaultdict(set)
    examples, sentence_local = {}, {}
    primary_words, primary_word_docs = Counter(), defaultdict(set)
    ngram_counts = {n:Counter() for n in (3,4,5,6)}
    ngram_docs = {n:defaultdict(set) for n in (3,4,5,6)}
    phrase_rows, phrase_example = [], {}
    for cap in CAPS:
        corpus = []
        for path in sorted((SOURCE/f"budget_{cap}/shards").glob("row_*.json")):
            row = read(path)
            corpus.append(row)
            provenance.append(dict(path=path.relative_to(SOURCE).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            text = normalize(row["thinking_text"])
            words = WORDS.findall(text)
            parts = list(sentence_parts(row["thinking_text"]))
            counts = Counter(key for key,_,_ in parts)
            word_sizes = {key:n for key,_,n in parts}
            sentence_words = sum(n for _,_,n in parts)
            repeated_excess = sum((count-1)*word_sizes[key] for key,count in counts.items())
            top_sentence, top_count = counts.most_common(1)[0]
            metric = dict(row_id=row["row_id"],budget=cap,hop=row["computed_hop"],correct=row["correct"],parsed=row["parse_success"],truncated=row["hit_max_new_tokens"],thinking_tokens=row["thinking_token_count"],word_count=len(words),sentence_count=len(parts),unique_sentences=len(counts),extra_repeated_sentence_words=repeated_excess,repeated_sentence_word_fraction=repeated_excess/len(words) if words else 0,most_repeated_sentence=top_sentence,most_repeated_sentence_count=top_count)
            row_metrics.append(metric)
            if cap!=32768:
                continue
            sentence_local[row["row_id"]] = counts
            for key,example,n in parts:
                if key not in examples:
                    examples[key]=example
            primary_sentence_counts.update(counts)
            for key in counts:
                primary_sentence_docs[key].add(row["row_id"])
            content_words = [w for w in words if len(w)>1 and w not in STOP and not w[0].isdigit()]
            primary_words.update(content_words)
            for w in set(content_words):
                primary_word_docs[w].add(row["row_id"])
            # N-grams stay within each sentence/newline part; no cross-boundary phrases.
            for key,_,_ in parts:
                tokens = WORDS.findall(key)
                for n in ngram_counts:
                    grams = Counter(" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1))
                    ngram_counts[n].update(grams)
                    for gram in grams:
                        ngram_docs[n][gram].add(row["row_id"])
        assert len(corpus)==100 and len({r["row_id"] for r in corpus})==100
        corpora[cap]=corpus
        all_text = [normalize(r["thinking_text"]) for r in corpus]
        total_words = sum(len(WORDS.findall(t)) for t in all_text)
        for phrase,(pattern,meaning) in PHRASES.items():
            regex = re.compile(pattern)
            counts=[len(regex.findall(t)) for t in all_text]
            phrase_rows.append(dict(budget=cap,phrase=phrase,meaning=meaning,total_occurrences=sum(counts),trajectories_with_phrase=sum(c>0 for c in counts),occurrences_per_1000_words=sum(counts)/total_words*1000,max_in_one_trace=max(counts),row_of_max=corpus[counts.index(max(counts))]["row_id"]))
            if cap==32768:
                selected = sorted(zip(corpus,all_text,counts),key=lambda item:(-item[2],item[0]["row_id"]))[0]
                row,text,count=selected; match=regex.search(text)
                phrase_example[phrase]=dict(row_id=row["row_id"],count=count,example=text[max(0,match.start()-100):match.end()+180] if match else "")

    primary = [r for r in row_metrics if r["budget"]==32768]
    sentence_rows=[]
    for key,count in primary_sentence_counts.most_common():
        if len(WORDS.findall(key))<4:
            continue
        docs=sorted(primary_sentence_docs[key])
        locals_={rid:sentence_local[rid][key] for rid in docs}
        rid=max(locals_,key=locals_.get)
        sentence_rows.append(dict(sentence=examples[key],normalized_sentence=key,total_occurrences=count,trajectories=len(docs),row_ids=";".join(map(str,docs)),max_in_one_trace=locals_[rid],row_of_max=rid))
        if len(sentence_rows)>=200:
            break
    grams=[]
    for n in ngram_counts:
        broad=[(gram,count,len(ngram_docs[n][gram])) for gram,count in ngram_counts[n].items() if len(ngram_docs[n][gram])>=5]
        for gram,count,docs in sorted(broad,key=lambda x:(-x[2],-x[1],x[0]))[:100]:
            grams.append(dict(n=n,phrase=gram,total_occurrences=count,trajectories=docs))
    word_rows=[dict(word=w,total_occurrences=count,trajectories=len(primary_word_docs[w])) for w,count in primary_words.most_common(150)]
    summaries=[]
    for cap in CAPS:
        metrics=[r for r in row_metrics if r["budget"]==cap]
        for group in ("all","correct","wrong_parsed","truncated"):
            rs=[r for r in metrics if group=="all" or (group=="correct" and r["correct"]) or (group=="wrong_parsed" and r["parsed"] and not r["correct"]) or (group=="truncated" and r["truncated"])]
            if not rs:
                continue
            words=sum(r["word_count"] for r in rs)
            summaries.append(dict(budget=cap,group=group,n=len(rs),thinking_tokens=sum(r["thinking_tokens"] for r in rs),words=words,extra_repeated_sentence_words=sum(r["extra_repeated_sentence_words"] for r in rs),weighted_repeated_sentence_word_fraction=sum(r["extra_repeated_sentence_words"] for r in rs)/words,median_trace_repeat_fraction=statistics.median(r["repeated_sentence_word_fraction"] for r in rs),max_sentence_repetitions=max(r["most_repeated_sentence_count"] for r in rs)))
    main_summary=next(s for s in summaries if s["budget"]==32768 and s["group"]=="all")
    truncated=next(s for s in summaries if s["budget"]==32768 and s["group"]=="truncated")
    result=dict(scope="100 unique 32K-cap visible thinking traces as primary corpus; separate 8K/16K paired diagnostics; no pooling of shared budget prefixes",primary_rows=100,all_budget_records=300,only_field="thinking_text; excludes prompt and final_response_text",sentence_normalization="casefold, curly apostrophe normalization, whitespace normalization, strip outer punctuation; retain entity letters and internal coordinate numbers",sentence_segmentation="split at terminal punctuation followed by whitespace or at newlines; heuristic, no semantic claim",word_tokenizer=WORDS.pattern,ngram_boundary="within sentence/newline parts only",repetition_metric="extra occurrences after first exact normalized sentence occurrence, weighted by sentence word count / all thinking words; descriptive exact repetition, not a semantic validity label",stopwords=sorted(STOP),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),primary=main_summary,truncated=truncated,truncated_token_share=truncated["thinking_tokens"]/main_summary["thinking_tokens"],truncated_repeat_word_share=truncated["extra_repeated_sentence_words"]/main_summary["extra_repeated_sentence_words"],groups=summaries,phrases=[r for r in phrase_rows if r["budget"]==32768],phrase_examples=phrase_example,top_repeated_sentences=sentence_rows[:25],top_broad_ngrams=[r for r in grams if r["n"] in (4,5,6)][:100],source_provenance=provenance)
    (OUT/"frequency_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    writecsv("phrase_frequencies.csv",phrase_rows)
    writecsv("word_frequencies_32k.csv",word_rows)
    writecsv("ngram_frequencies_32k.csv",grams)
    writecsv("repeated_sentences_32k.csv",sentence_rows)
    writecsv("per_trajectory_repetition.csv",row_metrics)
    print(json.dumps(dict(primary=main_summary,truncated=truncated,truncated_token_share=result["truncated_token_share"],truncated_repeat_word_share=result["truncated_repeat_word_share"],phrases=result["phrases"],top_repeated_sentences=sentence_rows[:10],broad_ngrams=[r for r in grams if r["n"]==5][:15],top_words=word_rows[:15]),ensure_ascii=True),flush=True)


if __name__=="__main__":
    main()
