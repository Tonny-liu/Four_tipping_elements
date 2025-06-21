import matplotlib.pyplot as plt
import models_simulation as ms


c_list = ['#d13559','grey','#1A6FDE','green','#e1a14d','#006e5f']

fig, axs = plt.subplots(nrows=5, ncols=4,sharex='col', figsize=(20, 14))

ms.make_case1(fig,axs[:,0],c_list)
ms.make_case2(fig,axs[:,1],c_list)
ms.make_case3(fig,axs[:,2],c_list)
ms.make_case4(fig,axs[:,3],c_list)


font_legend = {'family': 'Times New Roman', 'size': 25}
axs[4,0].set_xlabel(r"Time (a.u.)",**font_legend)
axs[4,1].set_xlabel(r"Time (a.u.)",**font_legend)
axs[4,2].set_xlabel(r"Time (a.u.)",**font_legend)
axs[4,3].set_xlabel(r"Time (a.u.)",**font_legend)
axs[0,0].set_ylabel(r"$x$",**font_legend)
axs[1,0].set_ylabel(r"$y$",**font_legend)
axs[2,0].set_ylabel("AC1",**font_legend)
axs[3,0].set_ylabel("Variance",**font_legend)
axs[4,0].set_ylabel(r"$\lambda$",**font_legend)

bwith = 1.0
ax_list = axs.flatten()
for i in range(len(ax_list)):
    ms.ax_set_homemade(ax_list[i],bwith)
    ax_list[i].tick_params(bottom=True, left=True, axis='both', which='major', width=1.0, length=4, labelsize=19)

axs[0,0].spines['top'].set_linewidth(bwith)
axs[0,0].spines['right'].set_linewidth(bwith)
axs[0,1].spines['top'].set_linewidth(bwith)
axs[0,1].spines['right'].set_linewidth(bwith)
axs[0,2].spines['top'].set_linewidth(bwith)
axs[0,2].spines['right'].set_linewidth(bwith)
axs[0,3].spines['top'].set_linewidth(bwith)
axs[0,3].spines['right'].set_linewidth(bwith)



for i in [2,3,4]:
    for k in [0,1,2,3]:
        ms.change_axis_scale(axs[i,k])


# ms.change_axis_scale(axs[2,0])
# ms.change_axis_scale(axs[2,1])
# ms.change_axis_scale(axs[2,2])
# ms.change_axis_scale(axs[2,3])
# ms.change_axis_scale(axs[3,0])
# ms.change_axis_scale(axs[3,1])
# ms.change_axis_scale(axs[3,2])
# ms.change_axis_scale(axs[3,3])
# ms.change_axis_scale(axs[4,0])
# ms.change_axis_scale(axs[4,1])
# ms.change_axis_scale(axs[4,2])
# ms.change_axis_scale(axs[4,3])


ms.make_subplot_label(axs)

plt.tight_layout()
plt.savefig("./cascading_models.pdf", bbox_inches='tight')

plt.show()
plt.close()