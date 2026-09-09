import json, numpy as np, trimesh, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT="/scratch2/LiteReality-simready/assets/sim-ready"
R="/scratch2/LiteReality-Agent/run"
OBJS=[("MIL-Meeting-Zhening","Wall4_Door_0","door · 1 hinge"),
      ("fallside-kitchen-Zhening","Dishwasher0","dishwasher · hinge + slide"),
      ("Kitchen-Xiaoyang_Lyu","Refrigerator0","fridge · 4 doors"),
      ("Airbnb-Cam-Zhening","Storage1","storage · 6 joints"),
      ("Airbnb-Room-Zhening","Bed0","bed · 4 drawers")]
sd=lambda s,o: f"{R}/{s}/scene_init/obj_stage/reconstructed_objs/{o}/sim"
PAL=plt.get_cmap("tab20").colors

def rodrigues(v, axis, ang):
    a=np.asarray(axis,float); a=a/np.linalg.norm(a); c,s=np.cos(ang),np.sin(ang)
    return v*c + np.cross(a,v)*s + np.outer(v@a, a)*(1-c)

def chain_origin(m, link):
    off=np.zeros(3); cur=link
    for _ in range(8):
        j=next((j for j in m["joints"] if j["child"]==cur), None)
        if j is None: break
        off=off+np.array(j["origin"],float); cur=j["parent"]
    return off

def posed(m, d, opened):
    """(vertices, faces, is_moving) per collider, in object coordinates."""
    out=[]
    for link in m["links"]:
        j=next((j for j in m["joints"] if j["child"]==link["name"]), None)
        base=chain_origin(m, link["name"])
        for col in link["colliders"]:
            mesh=trimesh.load(f"{d}/{col['file']}", process=False)
            v=np.asarray(mesh.vertices,float)
            if j is not None and opened:
                if j["type"]=="prismatic":
                    v=v+np.array(j["axis"],float)*j["limit_upper"]
                else:
                    v=rodrigues(v, j["axis"], j["limit_upper"])
            out.append((v+base, mesh.faces, j is not None))
    return out

fig=plt.figure(figsize=(19.5,8.6))
for k,(scan,obj,sub) in enumerate(OBJS):
    d=sd(scan,obj); m=json.load(open(f"{d}/{obj}.physics.json"))
    ncol=sum(len(l["colliders"]) for l in m["links"])
    for row,opened in ((0,False),(1,True)):
        ax=fig.add_subplot(2,5,row*5+k+1,projection="3d")
        V=[]
        for i,(v,f,moving) in enumerate(posed(m,d,opened)):
            V.append(v)
            face = PAL[(i*3)%20] if not moving else ("#e8663a" if opened else "#f0b8a4")
            ax.add_collection3d(Poly3DCollection(v[f], alpha=.62 if moving else .42,
                                facecolor=face, edgecolor="k", linewidths=.12))
        V=np.vstack(V); lo,hi=V.min(0),V.max(0); c=(lo+hi)/2; r=(hi-lo).max()/2*1.02
        ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
        ax.set_box_aspect((1,1,1)); ax.view_init(elev=16,azim=-62); ax.set_axis_off()
        if row==0:
            ax.set_title(f"{obj}\n{sub}\n{ncol} convex colliders · {m['total_mass']:.1f} kg",
                         fontsize=9.5,pad=-4)
        else:
            lims=", ".join(f"{j['limit_upper']:.2f}" for j in m["joints"][:4])
            ax.set_title(f"open at limit  ({lims}{' …' if len(m['joints'])>4 else ''})",
                         fontsize=8.5,pad=-4)
fig.text(.5,.975,"What the solver sees — convex decomposition (top, closed) and the same colliders "
         "driven to each joint's limit (bottom, moving parts in orange)",
         ha="center",fontsize=13)
fig.subplots_adjust(left=.01,right=.99,top=.93,bottom=.01,wspace=.02,hspace=.02)
fig.savefig(f"{OUT}/colliders.png",dpi=115); plt.close(fig); print("colliders.png")

fig,(a1,a2)=plt.subplots(1,2,figsize=(13.5,4.5))
for k,(scan,obj,sub) in enumerate(OBJS):
    d=sd(scan,obj)
    mo=mujoco.MjModel.from_xml_path(f"{d}/{obj}_drop.xml"); da=mujoco.MjData(mo)
    mujoco.mj_forward(mo,da); ts,zs=[],[]
    for i in range(int(1.0/mo.opt.timestep)):
        mujoco.mj_step(mo,da); ts.append(i*mo.opt.timestep)
        zs.append(float(np.asarray(da.xpos[1:])[:,2].min()))
    a1.plot(ts,np.array(zs)*1000,lw=1.6,color=PAL[k*2],label=obj)
a1.axhline(0,color="k",lw=.8,ls="--"); a1.set_xlabel("time (s)")
a1.set_ylabel("lowest body origin (mm)"); a1.grid(alpha=.25)
a1.set_title("Drop — released 50 mm up; all five settle within ~0.2 s",fontsize=11)
a1.text(.42,.42,"resting heights differ because a body origin sits on its own joint,\nnot on the floor — penetration is measured separately (max 0.03 mm)",
        transform=a1.transAxes,fontsize=7.5,color="#555")
a1.legend(fontsize=8,frameon=False)
names,meas,pred=[],[],[]
for scan,obj,_ in OBJS:
    c=json.load(open(f"{sd(scan,obj)}/{obj}.sim_check.json"))["scenarios"]["tilt"]
    names.append(obj); meas.append(c["measured_slide_deg"]); pred.append(c["expected_slide_deg"])
x=np.arange(len(names))
a2.bar(x-.19,pred,.38,label="predicted from stated friction · atan(µ)",color="#9aa7b8")
a2.bar(x+.19,meas,.38,label="measured slide angle",color="#2f6f9f")
a2.set_xticks(x); a2.set_xticklabels(names,rotation=18,ha="right",fontsize=8.5)
a2.set_ylabel("degrees"); a2.grid(axis="y",alpha=.25); a2.legend(fontsize=8,frameon=False)
a2.set_title("Tilt — does it slide at the angle its own friction claims?",fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/physics.png",dpi=115); plt.close(fig); print("physics.png")
