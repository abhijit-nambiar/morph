"""Small, explicit, offline vocabulary for the independent soft target."""
import re

# phrase -> (exclusive category, independent target-control changes)
PHRASES={
    'warm':('temperature',{'tone_temperature':.12}),
    'cool':('temperature',{'tone_temperature':-.12}),
    'brighter':('exposure',{'tone_exposure':.2}),
    'darker':('exposure',{'tone_exposure':-.2}),
    'soft contrast':('contrast',{'tone_contrast':-.2}),
    'strong contrast':('contrast',{'tone_contrast':.2}),
    'lift shadows':('shadows',{'tone_shadows':.2}),
    'deepen shadows':('shadows',{'tone_shadows':-.2}),
    'protect highlights':('highlights',{'tone_highlights':-.2}),
    'brighter highlights':('highlights',{'tone_highlights':.2}),
    'muted blues':('blue',{'cool_saturation':-.25}),
    'vivid blues':('blue',{'cool_saturation':.25}),
    'muted warm colors':('warmcolors',{'warm_saturation':-.25}),
    'vivid warm colors':('warmcolors',{'warm_saturation':.25}),
}

# Explicit equivalents only: reject the rest rather than guessing user intent.
ALIASES={
    'warmer':'warm', 'warm up':'warm', 'cooler':'cool', 'cool down':'cool',
    'brighten':'brighter', 'increase exposure':'brighter',
    'darken':'darker', 'reduce exposure':'darker',
    'lower contrast':'soft contrast', 'reduce contrast':'soft contrast',
    'increase contrast':'strong contrast',
    'brighten shadows':'lift shadows', 'open shadows':'lift shadows',
    'darken shadows':'deepen shadows',
    'reduce highlights':'protect highlights', 'lower highlights':'protect highlights',
    'brighten highlights':'brighter highlights',
    'desaturate blues':'muted blues', 'saturate blues':'vivid blues',
    'desaturate warm colors':'muted warm colors', 'saturate warm colors':'vivid warm colors',
}

def parse(prompt):
    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Prompt must not be empty')
    text=prompt.lower().strip(); text=re.sub(r'\band\b',',',text); clauses=[x.strip() for x in re.split(r'[,;]',text)]
    if not clauses or any(not x for x in clauses):raise ValueError('Use comma-separated supported phrases without empty clauses')
    result={};seen={};interpreted=[]
    for clause in clauses:
        intensity=1.; phrase=clause
        for prefix,factor in (('slightly ',.5),('strongly ',1.5)):
            if phrase.startswith(prefix):intensity=factor;phrase=phrase[len(prefix):];break
        phrase=ALIASES.get(phrase,phrase)
        if phrase not in PHRASES:raise ValueError('Unsupported phrase: '+repr(clause)+'. Supported phrases: '+', '.join(PHRASES))
        category,changes=PHRASES[phrase]
        if category in seen:raise ValueError('Conflicting or repeated prompt instructions: '+seen[category]+' and '+clause)
        seen[category]=clause
        for k,v in changes.items():result[k]=result.get(k,0.)+v*intensity
        interpreted.append({'phrase':phrase,'intensity':intensity})
    return result,interpreted

# Position names are frame-relative; they do not identify people or objects.
REGIONS={'upper left':(.2,.2),'upper center':(.5,.2),'upper right':(.8,.2),
         'middle left':(.2,.5),'center':(.5,.5),'middle right':(.8,.5),
         'lower left':(.2,.8),'lower center':(.5,.8),'lower right':(.8,.8)}
LOCAL_PHRASES={f'{action} {region}':(center,amount) for action,amount in (('brighten',.3),('darken',-.3)) for region,center in REGIONS.items()}

def parse_with_local(prompt):
    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Prompt must not be empty')
    clauses=[x.strip() for x in re.split(r'[,;]',re.sub(r'\band\b',',',prompt.lower().strip()))]
    if any(not clause for clause in clauses):raise ValueError('Use comma-separated supported phrases without empty clauses')
    global_clauses=[];local=None;local_interpretation=None
    for clause in clauses:
        phrase=clause;intensity=1.
        for prefix,factor in (('slightly ',.5),('strongly ',1.5)):
            if phrase.startswith(prefix):intensity=factor;phrase=phrase[len(prefix):];break
        if phrase in LOCAL_PHRASES:
            if local is not None:raise ValueError('Only one local lighting instruction is supported')
            center,amount=LOCAL_PHRASES[phrase];local={'center':list(center),'strength':amount*intensity}
            local_interpretation={'phrase':phrase,'intensity':intensity,'center':list(center),'local_strength':amount*intensity}
        else:global_clauses.append(clause)
    settings,interpreted=parse(', '.join(global_clauses)) if global_clauses else ({},[])
    if local_interpretation:interpreted.append(local_interpretation)
    return settings,interpreted,local
