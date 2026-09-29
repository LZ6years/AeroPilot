"""Render V2 offline artifacts without another model call or changing source logs."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import html
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from plugins.research.flight_experience.calls import write
from plugins.research.flight_experience.contracts import validate,prompt_manifest


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build(root):
    root=root.resolve()
    def esc(value):
        return html.escape(str(value))
    def link(path,label):
        return '<a href="'+esc(os.path.relpath(path,root).replace('\\','/'))+'">'+esc(label)+'</a>'
    def dump(value):
        return '<pre>'+esc(json.dumps(value,ensure_ascii=False,indent=2))+'</pre>'
    def detail(title,value):
        return '<details><summary>'+esc(title)+'</summary>'+dump(value)+'</details>'
    def card(value):
        body='<article><h3>'+esc(value.get('title',value.get('id','经验')))+'</h3>'
        for key,label in [('id','ID'),('decision_question','决策问题'),('applicability','适用条件'),
                ('observed_evidence','观察依据（模型表述）'),('recommendation','建议'),('limitations','局限')]:
            if key in value:
                body+='<p><b>'+label+'：</b>'+esc(value[key])+'</p>'
        return body+detail('全部字段与证据引用',value)+'</article>'
    rows=[]
    for path in sorted(root.rglob('calls.json')):
        for record in load(path):
            row={k:record.get(k) for k in ('stage','status','elapsed_seconds','usage','category','error_type','prompt_sha256','transport')}
            row['audit']=str(path.relative_to(root)).replace('\\','/')
            row['format_pass']=None
            if 'response_text' in record:
                try:
                    schema=record['caller_request']['response_format']['json_schema']['schema']
                    validate(json.loads(record['response_text']),schema)
                    row['format_pass']=True
                except (ValueError,TypeError,KeyError):
                    row['format_pass']=False
            rows.append(row)
    active=[r for r in rows if r['status']!='reused']
    known=[r for r in active if isinstance(r['usage'],dict)]
    tokens={k:sum(r['usage'].get(k,0) for r in known) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    tokens['cached_tokens']=sum((r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0) for r in known)
    tokens['reasoning_tokens']=sum((r['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens',0) for r in known)
    summary={'logical_calls':len(active),'returned':sum(r['status']=='received' for r in active),
        'timeouts':sum(r['category']=='timeout' for r in active),'missing_usage':sum(r['usage'] is None for r in active),
        'format_valid_returned':sum(r['format_pass'] is True for r in active),
        'format_invalid_returned':sum(r['format_pass'] is False for r in active),
        'format_repairs':sum('format_repair' in r['stage'] for r in active),
        'logical_call_seconds':sum(r['elapsed_seconds'] or 0 for r in active),
        'usage':tokens,'cost_usd':None,'calls':rows}
    summary['cohorts']={}
    retries=[retry for row in active for retry in (row.get('transport') or {}).get('retries',[])]
    summary['reported_transport_retries']=dict(Counter(r.get('category','unknown') for r in retries))
    summary['missing_transport_trace']=sum(not row.get('transport') for row in active)
    for name in ('capability','comparison/v1','comparison/v2','segments','followup'):
        group=[r for r in active if r['audit'].startswith(name+'/')]
        summary['cohorts'][name]={'calls':len(group),'returned':sum(r['status']=='received' for r in group),
            'timeouts':sum(r['category']=='timeout' for r in group),
            'known_total_tokens':sum((r['usage'] or {}).get('total_tokens',0) for r in group),
            'missing_usage':sum(r['usage'] is None for r in group)}
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>经验 V2 离线结果</title>',
        '<style>body{max-width:1120px;margin:36px auto;padding:0 24px;font:16px/1.65 system-ui;color:#243247;background:#f5f7fb}h1,h2{color:#163c66}article,details{background:white;border:1px solid #d8e1ea;border-radius:8px;padding:16px;margin:12px 0}table{width:100%;border-collapse:collapse;background:white}td,th{border:1px solid #d8e1ea;padding:8px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}img{width:200px;margin:6px}a{color:#145ab0}</style>',
        '<h1>经验生成 V2：原始输出、筛选与成本</h1>',
        '<p>本报告只读取实验产物。未执行新飞行，插件默认关闭，实验来源不能进入 production。审核通过不等于建议正确或闭环收益。</p>',
        '<h2>汇总口径</h2><p>调用完成表示返回响应；格式通过、证据通过和内容通过分开统计。超时缺失 usage 不按零计费，金额未知。180 秒超时、最多一次超时重试；超时审计未提供传输尝试明细时，不将配置次数当实测次数。</p>',
        dump({k:v for k,v in summary.items() if k!='calls'})]
    parts.append('<table><tr><th>实验 / 阶段</th><th>状态</th><th>秒</th><th>格式</th><th>input / output</th></tr>')
    for row in rows:
        usage=row['usage'] or {}
        parts.append('<tr><td>'+link(root/row['audit'],row['audit']+' : '+row['stage'])+'</td><td>'+esc(row['status']+' / '+str(row['category']))+'</td><td>'+str(round(row['elapsed_seconds'] or 0,2))+'</td><td>'+str(row['format_pass'])+'</td><td>'+esc(str(usage.get('prompt_tokens','未知'))+' / '+str(usage.get('completion_tokens','未知')))+'</td></tr>')
    parts.append('</table><h2>严格 Schema 与原图能力探测</h2>'+dump(load(root/'capability.json')))
    manifest=load(root/'comparison_manifest.json')
    parts.append('<h2>四条旧候选：固定组对照</h2><p>四条作为一组，同六张原图、事件、模型、180 秒超时、输出 Schema 和独立审核提示。不是四次逐条配对，也不是仅凭历史不同配置运行比较。对照输出不发布。</p>'+link(root/'comparison_manifest.json','冻结输入与散列'))
    for candidate in manifest['material']['experiences']:
        parts.append(card(candidate))
    for version in ('v1','v2'):
        directory=root/'comparison'/version
        parts.append('<h3>'+version+' 提示结果</h3>'+dump(load(directory/'result.json')))
        if (directory/'proposal.json').exists():
            proposal=load(directory/'proposal.json')
            parts.append(detail('变更提案全文',proposal))
            for out in proposal['outputs']:
                parts.append(card(out['body']))
        parts.append(link(directory/'calls.json','提示原文、真实请求、响应和审核'))
    parts.append('<p><b>人工核对：</b>旧提示得到 approve，但“升高使地标确认成功”的因果和身份结论仍不足以当作已证实事实。新提示合并提案遗漏旧引用 turn_0039，被校验拒绝，未调用内容审核。两者都不能证明提示优化已有收益；保留旧条件及反例仍需检查。</p>')
    parts.append('<h2>三个冻结片段及额外诊断</h2>'+link(root/'segments_manifest.json','原始片段清单与散列'))
    if (root/'followup_manifest.json').exists():
        parts.append('<p>原始三片段结束后，补充了图片必须同时列入 evidence_refs、不得虚构动作收益的提示约束。只对同一运动反馈片段做额外诊断，输入图片与候选上限相同，提示散列不同；不替换首次结果。该诊断的生成引用通过，但审核选片返回不存在的 source_id=material/c1，程序拒绝，未进入内容审核。传输 trace 记录一次连接重试和一次超时重试，最终返回用时约 331 秒。</p>'+link(root/'followup_manifest.json','额外诊断清单'))
    states=Counter();published=[];candidate_count=0;issues=Counter();revisions=0
    for path in sorted(list(root.glob('segments/*/experiment.json'))+list(root.glob('followup/*/experiment.json'))):
        result=load(path);directory=path.parent
        parts.append('<h3>'+esc(str(directory.relative_to(root)))+'</h3>'+dump({k:result.get(k) for k in ('status','error_type','category','publication_error','elapsed_seconds','published')}))
        parts.append(link(directory/'report.html','逐例报告与原图')+' · '+link(path,'完整候选与审核 JSON')+' · '+link(directory/'calls.json','实际调用与 usage'))
        published.extend(result.get('published',[]))
        revisions+=len(result.get('revisions',[]))
        for review in result['reviews']:
            issues.update(review.get('validator_issues',[]))
        for candidate in result['candidates']:
            candidate_count+=1
            reviews=[r for r in result['reviews'] if r['id']==candidate['id']]
            states[reviews[-1]['decision'] if reviews else '未完成审核']+=1
            parts.append(card(candidate)+detail('逐轮审核与引用校验',reviews)+detail('修订过程',[r for r in result.get('revisions',[]) if r['id']==candidate['id']]))
        media=list((directory/'media').glob('*'))
        if media:
            parts.append('<details><summary>本实验调用提供过的原图（散列文件名；观察 ID 配对见调用审计）</summary>')
            for image in media:
                relative=os.path.relpath(image,root).replace('\\','/')
                parts.append('<a href="'+esc(relative)+'"><img loading="lazy" src="'+esc(relative)+'" alt="'+esc(image.stem)+'"></a>')
            parts.append('</details>')
    summary.update(generated_candidates=candidate_count,final_candidate_states=dict(states),published=published,
        citation_validation_issues=dict(issues),semantic_revision_rounds=revisions)
    parts.append('<h2>候选最终状态</h2>'+dump({'candidates':candidate_count,'states':dict(states),'published':published}))
    parts.append('<p>结构化引用问题与修订轮次（同一候选可能有多个问题）：</p>'+dump({'issues':dict(issues),'revision_rounds':revisions}))
    parts.append('<h2>人工内容核对与限制</h2><ul><li>动作反馈片段：第一轮 c1 有具体决策价值，但图片引用缺失；c2 把画面中楼顶位置升级为可停止爬升的判断，需要独立几何证据，不能因措辞合理放行。</li><li>转向片段：三条候选都缺图片引用，未进入内容审核；不能据此统计为 reviewer discard。</li><li>节点片段：选择条件命中了提及节点的历史意图，不保证片段真正覆盖到达节点。输出中的“逐次缩短动作保证安全”和“固定高度确保安全”缺少可见结果支撑。片段覆盖不足与模型概括错误分开看。</li><li>证据校验是引用边界检查，不是物理真值判定；独立审核仍可能复述旧误判。本轮样本小，不能估计稳定通过率或闭环收益。</li></ul>')
    parts.append('<h2>实现与验证</h2><p>98 项相关自动回归通过（合成夹具，包括自动发布、撤回、版本冲突、恢复、格式修复、事务回滚、证据丢失、插件默认关闭、来源 ID 枚举）。真实候选不为了验证发布路径而放宽规则。</p>')
    parts.append(link(ROOT/'tests/manual/flight_experience_v2.md','操作文档')+' · '+link(ROOT/'plugins/research/flight_experience/README.md','模块边界'))
    parts.append('<h3>交付时提示词</h3><p>原始实验实际提示以各 calls.json 为准。后续补充不会改写旧请求；维护补充约束只完成自动回归，未再做相同组真实调用。额外诊断之后，选片 Schema 增加 source_id 与 observation_id 枚举并明确来源键，已完成自动测试，未继续真实付费调用。首次对照 v1 审计中的 prompt_version 沿用了 V2 调用器标识，实验 variant、实际 system 文本和散列才标明对照分组；入口现已修正该元数据标签。</p>')
    for path in sorted((ROOT/'plugins/research/flight_experience/prompts/v2').glob('*.txt')):
        parts.append('<details><summary>'+esc(path.name)+'</summary><pre>'+esc(path.read_text(encoding='utf-8'))+'</pre></details>')
    parts.append('</html>')
    write(root/'summary.json',summary)
    write(root/'delivery_manifest.json',{'prompt_hashes':prompt_manifest(),'files':{
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'plugins/research/flight_experience').rglob('*') if p.is_file() and '__pycache__' not in p.parts}})
    (root/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    return {k:v for k,v in summary.items() if k!='calls'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().output),ensure_ascii=False,indent=2))
