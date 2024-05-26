"""FIle used to append data for a new plot in the file where I store all the instructions"""

X = ['ff_entail_diff_instructblip_flant','ff_entail_diff_clip_flant','ff_entail_diff_llava']
Y1 = ['instructblip_flant real unconditioned r1 ent','clip_flant real unconditioned r1 ent','llava real unconditioned r1 ent']
Y2 = ['instructblip_flant real conditioned r1 ent','clip_flant real conditioned r1 ent','llava real conditioned r1 ent']
TITLES = ['DIFF(instructblip(F,R),instructblip(F,S)) vs DIFF(instructblip(Vu,R),instructblip(Vc,R))',
          'DIFF(clip_flant(F,R),clip_flant(F,S)) vs DIFF(clip_flant(Vu,R),clip_flant(Vc,R))',
          'DIFF(llava(F,R),llava(F,S)) vs DIFF(llava(Vu,R),llava(Vc,R))']
XLABELS = ['D(instructblip(F,R),instructblip(F,S))','D(clip_flant(F,R),clip_flant(F,S))','D(llava(F,R),llava(F,S))']
YLABELS = ['D(instructblip(Vu,R),instructblip(Vc,R))','D(clip_flant(Vu,R),clip_flant(Vc,R))','D(llava(Vu,R),llava(Vc,R))']
MODELS = ['instructblip_flant','clip_flant','llava']
COLORS = ['#07fa85','#07fa85','#07fa85']
XLIMS = [[-1,1],[-1,1],[-1,1]]
YLIMS = [[-1,1],[-1,1],[-1,1]]
FILENAMES = ['21_fd(r-s)_vd(u-c)r','22_fd(r-s)_vd(u-c)r','23_fd(r-s)_vd(u-c)r']  # n = 24


""" FILENAMES:
  n_Xaxis_Yaxis:   (n sequential incremental number)
  F = frame, V = video
  R = real, S = synthetic
  D(a-b) = difference of a - b
  C = conditioned, U = unconditioned
"""

inf = open('plots_instruction', 'a')

for x,y1,y2,title,xlable,ylable,model,color,xlim,ylim,filename in zip(X,Y1,Y2,TITLES,XLABELS,YLABELS,MODELS,COLORS,XLIMS,YLIMS,FILENAMES):
    inf.write(f'{x}|{y1}|{y2}|{title}|{xlable}|{ylable}|{model}|{color}|{xlim[0]},{xlim[1]}|{ylim[0]},{ylim[1]}|{filename}|\n')
