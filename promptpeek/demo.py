"""中文讲解用的 PROMPTPEEK 复现轨迹。

在仓库根目录运行：

    python3 -m promptpeek.demo
    python3 -m promptpeek.demo --skip-qwen

这里的「服务器」是一个 radix tree + LPM 调度模型，对应论文里 SGLang
的调度行为，不是 GPU 上的 SGLang / vLLM。Qwen3-4B-Base 只扮演攻击者
本地的候选 token 生成器（论文 Section IV-A1）。
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from promptpeek.attack import plant_victim, probe, recover
from promptpeek.candidates import AlphabetProposer, OracleProposer
from promptpeek.scheduler import ServingEngine

SHANGHAI = timezone(timedelta(hours=8))

# 图 5 / 图 6 的英文例句。每个词一个 token，方便对着论文讲。
WORDS = ["Imagine", "you", "are", "an", "a", "the", "IT", "expert"]
WORD_ID = {word: index + 1 for index, word in enumerate(WORDS)}
ID_WORD = {index: word for word, index in WORD_ID.items()}
DUMMY_WORD = 99
ID_WORD[DUMMY_WORD] = "%"


def _now() -> str:
    return datetime.now(SHANGHAI).isoformat(timespec="seconds")


def _show(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _piece(token: int, names: dict[int, str] | None = None) -> str:
    table = ID_WORD if names is None else names
    return table.get(token, str(token))


def _label_order(order: list[str], candidates: list[int], tag: str, names: dict[int, str] | None = None) -> list[str]:
    labels = []
    for req_id in order:
        if req_id.startswith(f"{tag}-pre"):
            labels.append("前哑元")
        elif req_id.startswith(f"{tag}-post"):
            labels.append("后哑元")
        elif req_id.startswith(f"{tag}-c"):
            index = int(req_id.removeprefix(f"{tag}-c"))
            labels.append(f"候选[{_piece(candidates[index], names)}]")
        else:
            labels.append(req_id)
    return labels


def _probe_view(result, candidates: list[int], tag: str, names: dict[int, str] | None = None) -> dict:
    labels = _label_order(result.order, candidates, tag, names)
    return {
        "hit_token": None if result.hit_token is None else _piece(result.hit_token, names),
        "promoted": [_piece(tok, names) for tok in result.promoted],
        "unique_hit": result.hit_token is not None,
        "order": labels,
        "batches": result.batches,
        "batch_match_lens": result.batch_match_lens,
        "batch_priorities": result.batch_priorities,
        "n_requests": result.n_requests,
        "n_candidate_requests": result.n_candidate_requests,
        "n_dummy_requests": result.n_dummy_requests,
    }


def figure6() -> dict:
    """图 5 / 图 6：一个 token 的命中与未命中。"""
    victim_words = ["Imagine", "you", "are", "an", "IT", "expert"]
    victim = [WORD_ID[word] for word in victim_words]
    prefix = victim[:3]
    truth = victim[3]
    hit_candidates = [WORD_ID["an"], WORD_ID["a"], WORD_ID["the"]]
    miss_candidates = [WORD_ID["a"], WORD_ID["the"], WORD_ID["IT"]]

    hit_engine = ServingEngine(policy="lpm", max_batch=4)
    plant_victim(hit_engine, "user", victim)
    hit = probe(hit_engine, "attacker", prefix, hit_candidates, DUMMY_WORD, tag="hit")

    miss_engine = ServingEngine(policy="lpm", max_batch=4)
    plant_victim(miss_engine, "user", victim)
    miss = probe(miss_engine, "attacker", prefix, miss_candidates, DUMMY_WORD, tag="miss")

    hit_view = _probe_view(hit, hit_candidates, "hit")
    miss_view = _probe_view(miss, miss_candidates, "miss")

    print("受害者 prompt:", " ".join(victim_words))
    print("攻击者已还原的前缀:", " ".join(victim_words[:3]))
    print("这一步的真实下一个 token:", victim_words[3])
    print("哑元 token: %    max_batch = 4，所以前哑元恰好占满正在运行的 batch")
    print()
    print("命中（候选 an / a / the）完成顺序:")
    print("  " + " → ".join(hit_view["order"]))
    print(f"  判定: 唯一提前的候选是 {hit_view['hit_token']}")
    print(f"  命中所在 batch 的匹配长度: {hit.batch_match_lens[1]}")
    print(f"  剩下的候选所在 batch 的匹配长度: {hit.batch_match_lens[-1]}")
    print("  读法: 4 是「Imagine you are」再加上一个已经在缓存里的 token；")
    print("        3 只有前缀本身。相等的匹配长度保持到达顺序，所以 an 夹在两批哑元中间。")
    print()
    print("未命中（候选 a / the / IT，都不是 an）完成顺序:")
    print("  " + " → ".join(miss_view["order"]))
    print("  判定: 没有候选排在后哑元前面。")
    return {
        "victim": victim_words,
        "prefix": victim_words[:3],
        "truth": victim_words[3],
        "dummy": "%",
        "max_batch": 4,
        "hit": hit_view,
        "miss": miss_view,
    }


def scenario2() -> dict:
    """场景 2：模板已知，秘密是性别和一个三位账号。"""
    _show("场景 2a：性别只有三个取值，一轮就够")
    gender_name = {1: "男", 2: "女", 3: "其他"}
    template = [50, 51]
    template_name = {50: "性别", 51: "="}
    secret_gender = [1]
    victim = template + secret_gender
    engine = ServingEngine(policy="lpm", max_batch=4)
    plant_victim(engine, "user", victim)
    gender_probe = probe(
        engine,
        "attacker",
        template,
        [1, 2, 3],
        dummy=90,
        tag="g",
    )
    names = {**gender_name, **template_name, 90: "哑元"}
    gender_view = _probe_view(gender_probe, [1, 2, 3], "g", names)
    print("已知模板: 性别=")
    print("真实填空: 男")
    print("攻击者把 {男, 女, 其他} 一次送出。完成顺序:")
    print("  " + " → ".join(gender_view["order"]))
    print(f"  判定: {gender_view['hit_token']}    请求数: {gender_probe.n_requests}")

    _show("场景 2b：账号后三位 705，字母表 0-9，每批 4 个候选")
    account_template = [60, 61, 62]
    account = [7, 0, 5]
    account_engine = ServingEngine(policy="lpm", max_batch=4)
    plant_victim(account_engine, "user", account_template + account)
    proposer = AlphabetProposer(alphabet=list(range(10)), dummy_pool=list(range(80, 100)))
    recovery = recover(
        account_engine,
        "attacker",
        account_template,
        proposer,
        max_new_tokens=3,
        k=4,
        max_rounds=4,
    )
    print("已知模板 token: 账号后三位（三个模板 token，内容攻击者已经知道）")
    print("真实秘密:", "".join(str(tok) for tok in account))
    print("还原结果:", "".join(str(tok) for tok in recovery.tokens))
    print("停止原因:", recovery.stopped_reason)
    print("总请求数（哑元 + 候选）:", recovery.requests)
    candidate_requests = sum(step.rounds * 4 for step in recovery.steps)
    print("其中候选请求:", candidate_requests, "  哑元请求:", recovery.requests - candidate_requests)
    print("论文 BMI 例子大约 60 次请求就能挖出性别、年龄、身高、体重。")
    print("下面这个三位数字例子的总请求数也是 60，但是我们把哑元也算进去了；")
    print("论文 Table II 的 Req. 列通常只数候选请求。不要把两个数字当成同一次实验。")
    steps = []
    for index, step in enumerate(recovery.steps):
        batches = []
        for item in step.probes:
            batches.append(
                {
                    "candidates": item.candidate_tokens,
                    "hit": item.hit_token,
                    "n_requests": item.n_requests,
                }
            )
        print(
            f"  第 {index + 1} 位真实值 {account[index]}: "
            f"还原 {step.token}，{step.rounds} 轮，{step.requests} 个请求，"
            f"各轮候选 {[item.candidate_tokens for item in step.probes]}"
        )
        steps.append(
            {
                "truth": account[index],
                "recovered": step.token,
                "rounds": step.rounds,
                "requests": step.requests,
                "batches": batches,
            }
        )
    return {
        "gender": {
            "template": "性别=",
            "truth": "男",
            "alphabet": ["男", "女", "其他"],
            "probe": gender_view,
        },
        "account": {
            "template": "账号后三位",
            "truth": "705",
            "recovered": "".join(str(tok) for tok in recovery.tokens),
            "stopped_reason": recovery.stopped_reason,
            "requests_total": recovery.requests,
            "requests_candidate": candidate_requests,
            "requests_dummy": recovery.requests - candidate_requests,
            "per_token_rounds": [step.rounds for step in recovery.steps],
            "steps": steps,
            "note": "总请求 60 = 5 次探测 × (4 前哑元 + 4 后哑元 + 4 候选)。论文 BMI 的约 60 次是另一套提示和另一套计数，这里只说明小字母表暴力枚举的请求量可以很小。",
        },
    }


def defenses() -> dict:
    """同一条单 token 探针，在每种防御下还是不是「恰好一个候选提前」。"""
    prefix = [WORD_ID[word] for word in ["Imagine", "you", "are", "an"]]
    truth = WORD_ID["IT"]
    victim = prefix + [truth, WORD_ID["expert"]]
    candidates = [truth, WORD_ID["a"], WORD_ID["the"]]
    rows = []

    configs = [
        ("LPM，M=1（论文默认）", dict(policy="lpm", lpm_margin=1, isolated=False)),
        ("按租户隔离 radix tree（cache_salt）", dict(policy="lpm", lpm_margin=1, isolated=True)),
        ("FCFS", dict(policy="fcfs", lpm_margin=1, isolated=False)),
        ("LPM 分桶 M=2，前缀长度 4，+1 不跨桶", dict(policy="lpm", lpm_margin=2, isolated=False)),
    ]
    print(f"受害者下一个 token 是 IT。已知前缀长度 {len(prefix)}（Imagine you are an）。")
    for name, kwargs in configs:
        engine = ServingEngine(max_batch=4, **kwargs)
        plant_victim(engine, "user", victim)
        result = probe(engine, "attacker", prefix, candidates, DUMMY_WORD, tag="d")
        view = _probe_view(result, candidates, "d")
        leaked = result.hit_token == truth
        print(f"- {name}")
        print(f"    提前的候选: {view['promoted'] or '无'}    唯一命中: {view['unique_hit']}    泄漏真实 token: {leaked}")
        rows.append(
            {
                "name": name,
                "leaked": leaked,
                "unique_hit": view["unique_hit"],
                "promoted": view["promoted"],
                "order": view["order"],
                "condition_broken": _condition_broken(name),
            }
        )

    short = [WORD_ID[word] for word in ["Imagine", "you", "are"]]
    boundary_engine = ServingEngine(policy="lpm", lpm_margin=2, max_batch=4)
    plant_victim(boundary_engine, "user", victim)
    boundary_candidates = [WORD_ID["an"], WORD_ID["a"], WORD_ID["the"]]
    boundary = probe(
        boundary_engine,
        "attacker",
        short,
        boundary_candidates,
        DUMMY_WORD,
        tag="b",
    )
    boundary_view = _probe_view(boundary, boundary_candidates, "b")
    print("- 同一个 M=2，前缀改成 3（Imagine you are）")
    print(f"    3//2 = 1，4//2 = 2，多一个 token 仍然换桶。判定: {boundary_view['hit_token']}")
    rows.append(
        {
            "name": "LPM M=2 但前缀长度 3，整除边界仍泄漏",
            "leaked": boundary.hit_token == WORD_ID["an"],
            "unique_hit": boundary_view["unique_hit"],
            "promoted": boundary_view["promoted"],
            "order": boundary_view["order"],
            "condition_broken": "没有打断。match_len // M 在 L ≡ M-1 (mod M) 时，+1 仍会升档。",
        }
    )

    rare_engine = ServingEngine(policy="lpm", max_batch=4)
    plant_victim(rare_engine, "user", victim)
    rare = recover(
        rare_engine,
        "attacker",
        prefix,
        OracleProposer(
            target=victim,
            alphabet=[WORD_ID["a"], WORD_ID["the"], WORD_ID["you"], WORD_ID["are"]],
            dummy_pool=[97, 96, 95],
        ),
        max_new_tokens=1,
        k=4,
        max_rounds=3,
    )
    print(f"- 真实 token IT 不在候选字母表里: 还原 {rare.tokens}，原因 {rare.stopped_reason}，请求 {rare.requests}")
    rows.append(
        {
            "name": "罕见 token：真值不在本地候选列表",
            "leaked": False,
            "unique_hit": False,
            "promoted": [],
            "stopped_reason": rare.stopped_reason,
            "requests": rare.requests,
            "condition_broken": "侧信道还在，但是候选生成失败。攻击在这个 token 停下，后面的前缀也断了。",
        }
    )

    evicted = _eviction_demo()
    rows.append(evicted["row"])
    return {"rows": rows, "eviction": evicted}


def _condition_broken(name: str) -> str:
    if "隔离" in name:
        return "条件 2 的共享被拆掉：攻击者的 radix tree 里没有受害者前缀，LPM 无从比较。"
    if name.startswith("FCFS"):
        return "条件 2 的顺序比特被拆掉：完成顺序等于到达顺序，多个候选同时排在后哑元前面，检测器拒绝把它当成命中。"
    if "不跨桶" in name:
        return "条件 2 被削弱：差 1 个 token 不再改变优先级。缓存仍然共享，只是这一跳的顺序不再泄露。"
    return "三条条件都在，攻击成立。"


def _eviction_demo() -> dict:
    victim = [1, 2, 3, 4, 5]
    tight = ServingEngine(policy="lpm", max_batch=4, capacity=len(victim))
    plant_victim(tight, "user", victim)
    evicting = probe(tight, "attacker", [1, 2, 3], [4, 6], dummy=99, tag="ev")
    evictions_when_victim_dropped = tight.cache.evictions
    # Measured before the follow-up probe. That later probe sends token 5
    # as a candidate and would write it back, which is not the victim's KV.
    match_after_eviction = tight.cache.match_len("user", victim)
    follow = probe(tight, "attacker", [1, 2, 3, 4], [5, 8], dummy=98, tag="nx")
    roomy = ServingEngine(policy="lpm", max_batch=4, capacity=10_000)
    plant_victim(roomy, "user", victim)
    control = probe(roomy, "attacker", [1, 2, 3], [4, 6], dummy=99, tag="ok")
    print("- 缓存容量只够放下受害者：探测一开始就把受害者挤掉")
    print(
        f"    容量内对照命中 {control.hit_token}；"
        f"挤掉时命中 {evicting.hit_token}，第一次探测触发的驱逐次数 {evictions_when_victim_dropped}；"
        f"下一步再探 {follow.hit_token}"
    )
    return {
        "row": {
            "name": "KV 被驱逐（容量只够受害者自己）",
            "leaked": False,
            "unique_hit": evicting.hit_token is not None,
            "promoted": evicting.promoted,
            "evictions": evictions_when_victim_dropped,
            "control_hit": control.hit_token,
            "followup_hit": follow.hit_token,
            "condition_broken": "条件 1 被拆掉：受害者前缀已经不在 radix tree 里，下一次探测没有唯一命中。",
        },
        "control_hit": control.hit_token,
        "evicting_hit": evicting.hit_token,
        "followup_hit": follow.hit_token,
        "evictions_when_victim_dropped": evictions_when_victim_dropped,
        "match_len_after_evicting_probe": match_after_eviction,
        "victim_len": len(victim),
    }


def _qwen_sentence(proposer, text: str, hold: int = 2) -> dict:
    ids = proposer.tokenize(text)
    pieces = [proposer.piece(token) for token in ids]
    if len(ids) <= hold + 1:
        raise RuntimeError(f"句子太短，无法扣住后缀: {pieces}")
    known = ids[:-hold]
    secret = ids[-hold:]
    per_token = []
    prefix = list(known)
    for truth in secret:
        top = proposer._proposer.topk_ids(prefix, 8)
        per_token.append(
            {
                "truth": proposer.piece(truth),
                "truth_id": truth,
                "in_top8": truth in top,
                "top8": [proposer.piece(token) for token in top],
            }
        )
        prefix.append(truth)
    return {
        "text": text,
        "pieces": pieces,
        "known_text": proposer.detokenize(known),
        "secret_text": proposer.detokenize(secret),
        "known_ids": known,
        "secret_ids": secret,
        "per_token_topk": per_token,
        "first_in_top8": per_token[0]["in_top8"],
    }


def qwen_continuation(model_path: str) -> dict:
    from promptpeek.candidates import QwenBaseProposer

    _show("Qwen3-4B-Base 只负责提出候选 token，调度器仍然是我们的 radix/LPM 模型")
    proposer = QwenBaseProposer(model_path, n_ctx=256, n_threads=4)
    print(f"GGUF: {model_path}  n_ctx={proposer.n_ctx}  n_threads={proposer.n_threads}  n_gpu_layers=0")
    sentences = [
        "The capital of France is Paris.",
        "北京是中国的首都。",
        "To be or not to be.",
    ]
    checked = [_qwen_sentence(proposer, text) for text in sentences]
    chosen = next((item for item in checked if item["first_in_top8"]), checked[0])
    print("三句短句里，第一个被扣住的 token 是否落在 Qwen top-8：")
    for item in checked:
        flag = "在" if item["first_in_top8"] else "不在"
        print(f"  {item['text']!r}  后缀 {item['secret_text']!r}  首 token {flag} top-8")
    print("实际拿去跑侧信道的句子:", chosen["text"])
    print("已知前缀:", repr(chosen["known_text"]))
    print("真实后缀:", repr(chosen["secret_text"]))

    def propose(prefix, k, exclude):
        return proposer.propose(prefix, k, exclude)

    engine = ServingEngine(policy="lpm", max_batch=4, capacity=100_000)
    plant_victim(engine, "user", chosen["known_ids"] + chosen["secret_ids"])
    recovery = recover(
        engine,
        "attacker",
        chosen["known_ids"],
        propose,
        max_new_tokens=len(chosen["secret_ids"]),
        k=8,
        max_rounds=2,
    )
    recovered_text = proposer.detokenize(recovery.tokens)
    print("侧信道还原:", repr(recovered_text))
    print("总请求:", recovery.requests, "  原因:", recovery.stopped_reason)
    steps = []
    for index, step in enumerate(recovery.steps):
        truth = chosen["secret_ids"][index]
        first_batch = []
        in_first = False
        if step.probes:
            first_batch = [proposer.piece(token) for token in step.probes[0].candidate_tokens]
            in_first = truth in step.probes[0].candidate_tokens
        piece = None if step.token is None else proposer.piece(step.token)
        print(
            f"  token {index + 1}: 真实 {proposer.piece(truth)!r}  "
            f"还原 {piece!r}  第一批含真值={in_first}  "
            f"轮数={step.rounds}  请求={step.requests}"
        )
        steps.append(
            {
                "truth": proposer.piece(truth),
                "recovered": piece,
                "truth_in_first_batch": in_first,
                "first_batch": first_batch,
                "rounds": step.rounds,
                "requests": step.requests,
                "hit": step.token == truth,
            }
        )
    return {
        "loaded": True,
        "model": model_path,
        "role": "candidate generator only; the server is the radix/LPM scheduler",
        "n_ctx": proposer.n_ctx,
        "quantization": "Q4_K_M GGUF",
        "selection_rule": "在三句里选用第一个「被扣住的首 token 落在 top-8」的句子；如果都没有，就用第一句并记录失败。",
        "survey": [
            {
                "text": item["text"],
                "known": item["known_text"],
                "secret": item["secret_text"],
                "pieces": item["pieces"],
                "first_in_top8": item["first_in_top8"],
                "per_token_topk": item["per_token_topk"],
            }
            for item in checked
        ],
        "chosen_text": chosen["text"],
        "known": chosen["known_text"],
        "secret": chosen["secret_text"],
        "recovered": recovered_text,
        "exact": recovery.tokens == chosen["secret_ids"],
        "stopped_reason": recovery.stopped_reason,
        "requests": recovery.requests,
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PROMPTPEEK 调度侧信道的讲解用复现")
    parser.add_argument(
        "--qwen",
        default="",
        help="Qwen3-4B-Base 的 GGUF 路径。默认：模型文件存在就用。",
    )
    parser.add_argument("--skip-qwen", action="store_true", help="只跑调度器，不加载 Qwen")
    parser.add_argument("--out", default="results/demo_report.json")
    args = parser.parse_args()

    model_path = ""
    if not args.skip_qwen:
        model_path = args.qwen or "models/Qwen3-4B-Base.Q4_K_M.gguf"
        if not Path(model_path).is_file():
            model_path = ""

    _show("图 6：命中会把恰好一个候选插到后哑元前面")
    figure = figure6()
    scenario = scenario2()
    _show("防御：同一条探针，哪一条攻击条件被拆掉")
    defense = defenses()

    report = {
        "generated_at": _now(),
        "timezone": "Asia/Shanghai",
        "paper": {
            "title": "I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving",
            "authors": "Guanlong Wu, Zheng Zhang, Yao Zhang, Weili Wang, Jianyu Niu, Ye Wu, Yinqian Zhang",
            "venue": "NDSS 2025",
            "doi": "10.14722/ndss.2025.241772",
            "stack": "SGLang + Llama-2-13B，单卡 A100 80GB",
            "summary_rates": {
                "whole_prompt": {"sr": "95%", "rr": "81%"},
                "input": {"sr": "99%", "rr": "99%"},
                "template": {"sr": "98%", "rr": "91%"},
            },
        },
        "reproduction": {
            "server": "本仓库的 radix tree + LPM/FCFS/隔离/M 分桶调度模型",
            "not_run": "没有 GPU，没有跑 SGLang 或 vLLM，没有复现论文 Table II 的端到端数字",
        },
        "figure6": figure,
        "scenario2": scenario,
        "defenses": defense,
        "qwen": None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if model_path:
        try:
            report["qwen"] = qwen_continuation(model_path)
        except Exception as exc:
            report["qwen"] = {
                "loaded": False,
                "model": model_path,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print("Qwen 没有跑起来:")
            print(report["qwen"]["traceback"])
        report["generated_at"] = _now()
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        report["qwen"] = {"loaded": False, "error": "未提供 GGUF，或调用时使用了 --skip-qwen"}
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("结果已写入", out)


if __name__ == "__main__":
    main()
