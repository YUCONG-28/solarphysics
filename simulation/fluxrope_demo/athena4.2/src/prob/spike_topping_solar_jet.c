#include "copyright.h"
/*============================================================================*/
/*! \file spike_topping_solar_jet.c
 *  \brief Staged 2.5D stratified open-field solar-jet problem.
 *
 * This problem implements the adiabatic v4 baseline: three-component MHD,
 * hydrostatic stratification, an open field plus buried 2D dipole, a uniform
 * guide field, line-tied lower-boundary convergence/shear, solar gravity, and
 * a diode upper boundary.  Thermal conduction, optically thin radiation, and
 * background heating are intentionally separate validation stages.
 */
/*============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"

#ifdef HYDRO
#error : spike_topping_solar_jet requires MHD
#endif

static Real g_code, t_ch, t_cor, y_tr, w_tr, p_base;
static Real b_open, guide_ratio, dipole_depth, null_height;
static Real drive_converge, drive_shear, drive_width;
static Real drive_start, drive_ramp, drive_hold, drive_end;
static Real eta_background, eta_anomalous, current_threshold;
static Real density_floor, pressure_floor;
static int drive_enabled;
static int first_call=1;

static Real current_z_sq(const GridS *pG,const int i,const int j,const int k)
{
  Real value=(pG->B2i[k][j][i]-pG->B2i[k][j][i-1])/pG->dx1
    -(pG->B1i[k][j][i]-pG->B1i[k][j-1][i])/pG->dx2;
  return SQR(value);
}

static Real div_b_sq(const GridS *pG,const int i,const int j,const int k)
{
  Real value=(pG->B1i[k][j][i+1]-pG->B1i[k][j][i])/pG->dx1
    +(pG->B2i[k][j+1][i]-pG->B2i[k][j][i])/pG->dx2;
  return SQR(value);
}

static Real velocity_x_sq(const GridS *pG,const int i,const int j,const int k)
{
  return SQR(pG->U[k][j][i].M1/pG->U[k][j][i].d);
}

static Real local_eta_value(const GridS *pG,const int i,const int j,const int k)
{
  int il=pG->is-nghost,iu=pG->ie+nghost;
  int jl=pG->js-nghost,ju=pG->je+nghost;
  int im=MAX(i-1,il),ip=MIN(i+1,iu);
  int jm=MAX(j-1,jl),jp=MIN(j+1,ju);
  Real db2dx=(pG->U[k][j][ip].B2c-pG->U[k][j][im].B2c)
    /((Real)(ip-im)*pG->dx1);
  Real db1dy=(pG->U[k][jp][i].B1c-pG->U[k][jm][i].B1c)
    /((Real)(jp-jm)*pG->dx2);
  Real excess=MAX(fabs(db2dx-db1dy)-current_threshold,0.0);
  return eta_background+(eta_anomalous-eta_background)
    *(1.0-exp(-SQR(excess)));
}

static Real temperature_profile(const Real y)
{
  return t_ch + 0.5*(t_cor-t_ch)*(1.0+tanh((y-y_tr)/w_tr));
}

static Real hydrostatic_primitive(const Real y)
{
  /*
   * Stable antiderivative of 1/T for
   * T=a+(b-a)(1+tanh((y-y_tr)/w_tr))/2.
   * The two algebraically equivalent branches avoid exp overflow.
   */
  Real a=t_ch,b=t_cor,u=2.0*(y-y_tr)/w_tr;
  Real coefficient=(a-b)/(a*b);
  if (u >= 0.0) {
    return 0.5*w_tr*(u/b+coefficient*
      (log(b)+log1p((a/b)*exp(-u))));
  }
  return 0.5*w_tr*(u/a+coefficient*
    (log(a)+log1p((b/a)*exp(u))));
}

static Real pressure_profile(const Real y)
{
  return p_base*exp(-g_code*
    (hydrostatic_primitive(y)-hydrostatic_primitive(0.0)));
}

static Real vector_potential(const Real x, const Real y)
{
  Real yp=y+dipole_depth;
  Real r2=x*x+yp*yp;
  Real moment=b_open*SQR(null_height+dipole_depth);
  return -b_open*x + moment*x/r2;
}

static Real gravity_potential(const Real x1, const Real x2, const Real x3)
{
  (void)x1;
  (void)x3;
  return g_code*x2;
}

static Real analytic_b1_face(const GridS *pG,int i,int j,int k)
{
  Real x,y,z,xf;
  cc_pos((GridS *)pG,i,j,k,&x,&y,&z);
  xf=x-0.5*pG->dx1;
  return (vector_potential(xf,y+0.5*pG->dx2)
    -vector_potential(xf,y-0.5*pG->dx2))/pG->dx2;
}

static Real analytic_b2_face(const GridS *pG,int i,int j,int k)
{
  Real x,y,z,yf;
  cc_pos((GridS *)pG,i,j,k,&x,&y,&z);
  yf=y-0.5*pG->dx2;
  return -(vector_potential(x+0.5*pG->dx1,yf)
    -vector_potential(x-0.5*pG->dx1,yf))/pG->dx1;
}

static Real driver_envelope(const Real time)
{
  if (!drive_enabled) return 0.0;
  if (time <= drive_start || time >= drive_end) return 0.0;
  if (time < drive_start+drive_ramp) {
    return 0.5*(1.0-cos(PI*(time-drive_start)/drive_ramp));
  }
  if (time <= drive_hold) return 1.0;
  return 0.5*(1.0+cos(PI*(time-drive_hold)/(drive_end-drive_hold)));
}

static void lower_line_tied(GridS *pG)
{
  int i,j,k,n;
  int is=pG->is,ie=pG->ie,js=pG->js,ks=pG->ks,ke=pG->ke;
  Real x,y,z,envelope,vx,vz,rho;
  Real pressure;
  for (k=ks; k<=ke; k++) {
    for (n=1; n<=nghost; n++) {
      j=js-n;
      for (i=is-nghost; i<=ie+nghost; i++) {
        cc_pos(pG,i,j,k,&x,&y,&z);
        pG->U[k][j][i]=pG->U[k][js][i];
        pressure=pressure_profile(y);
        rho=pressure/temperature_profile(y);
        rho=MAX(rho,density_floor);
        pressure=MAX(pressure,pressure_floor);
        pG->U[k][j][i].B1c=0.5*
          (analytic_b1_face(pG,i,j,k)+analytic_b1_face(pG,i+1,j,k));
        pG->U[k][j][i].B2c=0.5*
          (analytic_b2_face(pG,i,j,k)+analytic_b2_face(pG,i,j+1,k));
        pG->U[k][j][i].B3c=guide_ratio*b_open;
        pG->U[k][j][i].d=rho;
        envelope=driver_envelope(pG->time)*exp(-SQR(x/drive_width));
        vx=-drive_converge*tanh(x/drive_width)*envelope;
        vz= drive_shear*tanh(x/drive_width)*envelope;
        pG->U[k][j][i].M1=rho*vx;
        pG->U[k][j][i].M2=0.0;
        pG->U[k][j][i].M3=rho*vz;
#ifndef BAROTROPIC
        pG->U[k][j][i].E=pressure/Gamma_1+0.5*rho*(vx*vx+vz*vz)
          +0.5*(SQR(pG->U[k][j][i].B1c)
                +SQR(pG->U[k][j][i].B2c)
                +SQR(pG->U[k][j][i].B3c));
#endif
      }
      for (i=is-(nghost-1); i<=ie+nghost; i++)
        pG->B1i[k][j][i]=analytic_b1_face(pG,i,j,k);
      if (n<nghost) for (i=is-nghost; i<=ie+nghost; i++)
        pG->B2i[k][j][i]=analytic_b2_face(pG,i,j,k);
      for (i=is-nghost; i<=ie+nghost; i++) {
        pG->B3i[k][j][i]=pG->B3i[k][js][i];
      }
    }
  }
}

static void upper_diode(GridS *pG)
{
  int i,j,k,n;
  int is=pG->is,ie=pG->ie,je=pG->je,ks=pG->ks,ke=pG->ke;
  Real x,y,z,pressure,rho,vx,vy,vz;
  for (k=ks; k<=ke; k++) {
    for (n=1; n<=nghost; n++) {
      j=je+n;
      for (i=is-nghost; i<=ie+nghost; i++) {
        cc_pos(pG,i,j,k,&x,&y,&z);
        pG->U[k][j][i]=pG->U[k][je][i];
        vx=pG->U[k][j][i].M1/pG->U[k][j][i].d;
        vy=MAX(pG->U[k][j][i].M2/pG->U[k][j][i].d,0.0);
        vz=pG->U[k][j][i].M3/pG->U[k][j][i].d;
        pressure=pressure_profile(y);
        rho=pressure/temperature_profile(y);
        pG->U[k][j][i].d=rho;
        pG->U[k][j][i].M1=rho*vx;
        pG->U[k][j][i].M2=rho*vy;
        pG->U[k][j][i].M3=rho*vz;
#ifndef BAROTROPIC
        pG->U[k][j][i].E=pressure/Gamma_1
          +0.5*rho*(vx*vx+vy*vy+vz*vz)
          +0.5*(SQR(pG->U[k][j][i].B1c)
                +SQR(pG->U[k][j][i].B2c)
                +SQR(pG->U[k][j][i].B3c));
#endif
      }
      for (i=is-(nghost-1); i<=ie+nghost; i++)
        pG->B1i[k][j][i]=pG->B1i[k][je][i];
      if (n>1) for (i=is-nghost; i<=ie+nghost; i++)
        pG->B2i[k][j][i]=pG->B2i[k][je][i];
      for (i=is-nghost; i<=ie+nghost; i++) {
        pG->B3i[k][j][i]=pG->B3i[k][je][i];
      }
    }
  }
}

static void left_diode(GridS *pG)
{
  int i,j,k,n;
  int is=pG->is,js=pG->js,je=pG->je,ks=pG->ks,ke=pG->ke;
  for (k=ks; k<=ke; k++) for (j=js; j<=je; j++)
    for (n=1; n<=nghost; n++) {
      i=is-n;
      pG->U[k][j][i]=pG->U[k][j][is];
      if (pG->U[k][j][i].M1 > 0.0) pG->U[k][j][i].M1=0.0;
      if (n<nghost) pG->B1i[k][j][i]=pG->B1i[k][j][is];
      pG->B3i[k][j][i]=pG->B3i[k][j][is];
    }
  for (k=ks; k<=ke; k++) for (j=js; j<=je+1; j++)
    for (n=1; n<=nghost; n++)
      pG->B2i[k][j][is-n]=pG->B2i[k][j][is];
}

static void right_diode(GridS *pG)
{
  int i,j,k,n;
  int ie=pG->ie,js=pG->js,je=pG->je,ks=pG->ks,ke=pG->ke;
  for (k=ks; k<=ke; k++) for (j=js; j<=je; j++)
    for (n=1; n<=nghost; n++) {
      i=ie+n;
      pG->U[k][j][i]=pG->U[k][j][ie];
      if (pG->U[k][j][i].M1 < 0.0) pG->U[k][j][i].M1=0.0;
      if (n>1) pG->B1i[k][j][i]=pG->B1i[k][j][ie];
      pG->B3i[k][j][i]=pG->B3i[k][j][ie];
    }
  for (k=ks; k<=ke; k++) for (j=js; j<=je+1; j++)
    for (n=1; n<=nghost; n++)
      pG->B2i[k][j][ie+n]=pG->B2i[k][j][ie];
}

static void load_problem_parameters(void)
{
  g_code=par_getd_def("problem","gravity",0.12);
  t_ch=par_getd_def("problem","temperature_ch",0.013333333333);
  t_cor=par_getd_def("problem","temperature_cor",1.0);
  y_tr=par_getd_def("problem","transition_height",0.25);
  w_tr=par_getd_def("problem","transition_width",0.04);
  p_base=par_getd_def("problem","base_pressure",20.0);
  b_open=par_getd_def("problem","b_open",1.0);
  guide_ratio=par_getd_def("problem","guide_field_ratio",0.5);
  dipole_depth=par_getd_def("problem","dipole_depth",0.5);
  null_height=par_getd_def("problem","null_height",2.0);
  drive_enabled=par_geti_def("problem","drive_enabled",1);
  drive_converge=par_getd_def("problem","drive_converge",0.02);
  drive_shear=par_getd_def("problem","drive_shear",0.01);
  drive_width=par_getd_def("problem","drive_width",1.0);
  drive_start=par_getd_def("problem","drive_start",7.5);
  drive_ramp=par_getd_def("problem","drive_ramp",3.75);
  drive_hold=par_getd_def("problem","drive_hold",18.75);
  drive_end=par_getd_def("problem","drive_end",22.5);
  eta_background=par_getd_def("problem","eta_background",1.0e-5);
  eta_anomalous=par_getd_def("problem","eta_anomalous",2.0e-4);
  current_threshold=par_getd_def("problem","current_threshold",5.0);
  density_floor=par_getd_def("problem","density_floor",1.0e-10);
  pressure_floor=par_getd_def("problem","pressure_floor",1.0e-10);
  if (MIN(t_ch,t_cor)<=0.0 || p_base<=0.0 || b_open<=0.0 ||
      guide_ratio<0.0 || dipole_depth<=0.0 || null_height<=0.0 ||
      drive_width<=0.0 || drive_ramp<=0.0 || drive_end<=drive_hold ||
      density_floor<=0.0 || pressure_floor<=0.0 ||
      (drive_enabled!=0 && drive_enabled!=1)) {
    ath_error("[spike_topping_solar_jet]: invalid model parameter\n");
  }
#ifdef RESISTIVITY
  eta_Ohm=MAX(eta_background,eta_anomalous);
  Q_Hall=0.0;
  Q_AD=0.0;
#endif
#ifdef VISCOSITY
  nu_iso=par_getd_def("problem","nu_iso",1.0e-5);
  nu_aniso=0.0;
#endif
  StaticGravPot=gravity_potential;
}

static void register_problem(DomainS *pDomain)
{
  if (pDomain->Disp[0]==0) bvals_mhd_fun(pDomain,left_x1,left_diode);
  if (pDomain->MaxX[0]==pDomain->RootMaxX[0])
    bvals_mhd_fun(pDomain,right_x1,right_diode);
  if (pDomain->Disp[1]==0) bvals_mhd_fun(pDomain,left_x2,lower_line_tied);
  if (pDomain->MaxX[1]==pDomain->RootMaxX[1])
    bvals_mhd_fun(pDomain,right_x2,upper_diode);
}

void problem(DomainS *pDomain)
{
  GridS *pG=pDomain->Grid;
  int i,j,k;
  int is=pG->is,ie=pG->ie,js=pG->js,je=pG->je,ks=pG->ks,ke=pG->ke;
  int nx1=(ie-is)+1+2*nghost,nx2=(je-js)+1+2*nghost;
  int nx3=(ke-ks)+1+2*nghost;
  Real x,y,z,xf,yf,pressure,temp,rho;
  Real ***az;

  if ((je-js)==0 || (ke-ks)!=0) {
    ath_error("[spike_topping_solar_jet]: requires 2D grid with Nx3=1\n");
  }
  load_problem_parameters();

  if ((az=(Real***)calloc_3d_array(nx3,nx2,nx1,sizeof(Real)))==NULL) {
    ath_error("[spike_topping_solar_jet]: vector-potential allocation failed\n");
  }
  for (k=ks; k<=ke+1; k++) for (j=js; j<=je+1; j++)
    for (i=is; i<=ie+1; i++) {
      cc_pos(pG,i,j,k,&x,&y,&z);
      xf=x-0.5*pG->dx1;
      yf=y-0.5*pG->dx2;
      az[k][j][i]=vector_potential(xf,yf);
    }
  for (k=ks; k<=ke; k++) for (j=js; j<=je; j++)
    for (i=is; i<=ie+1; i++)
      pG->B1i[k][j][i]=(az[k][j+1][i]-az[k][j][i])/pG->dx2;
  for (k=ks; k<=ke; k++) for (j=js; j<=je+1; j++)
    for (i=is; i<=ie; i++)
      pG->B2i[k][j][i]=-(az[k][j][i+1]-az[k][j][i])/pG->dx1;
  for (k=ks; k<=ke; k++) for (j=js; j<=je; j++)
    for (i=is; i<=ie; i++) pG->B3i[k][j][i]=guide_ratio*b_open;

  for (k=ks; k<=ke; k++) for (j=js; j<=je; j++)
    for (i=is; i<=ie; i++) {
      cc_pos(pG,i,j,k,&x,&y,&z);
      pressure=pressure_profile(y);
      temp=temperature_profile(y);
      rho=pressure/temp;
      pG->U[k][j][i].d=rho;
      pG->U[k][j][i].M1=pG->U[k][j][i].M2=pG->U[k][j][i].M3=0.0;
      pG->U[k][j][i].B1c=0.5*(pG->B1i[k][j][i]+pG->B1i[k][j][i+1]);
      pG->U[k][j][i].B2c=0.5*(pG->B2i[k][j][i]+pG->B2i[k][j+1][i]);
      pG->U[k][j][i].B3c=guide_ratio*b_open;
#ifndef BAROTROPIC
      pG->U[k][j][i].E=pressure/Gamma_1
        +0.5*(SQR(pG->U[k][j][i].B1c)+SQR(pG->U[k][j][i].B2c)
              +SQR(pG->U[k][j][i].B3c));
#endif
    }
  free_3d_array((void***)az);
  register_problem(pDomain);
  if (first_call==1) {
    dump_history_enroll(current_z_sq,"<Jz2>");
    dump_history_enroll(div_b_sq,"<divB2>");
    dump_history_enroll(velocity_x_sq,"<vx2>");
    dump_history_enroll(local_eta_value,"<eta>");
    first_call=0;
  }
}

void problem_write_restart(MeshS *pM, FILE *fp) {(void)pM;(void)fp;}
void problem_read_restart(MeshS *pM, FILE *fp)
{
  int nl,nd;
  (void)fp;
  load_problem_parameters();
  for (nl=0; nl<pM->NLevels; nl++) {
    for (nd=0; nd<pM->DomainsPerLevel[nl]; nd++) {
      if (pM->Domain[nl][nd].Grid != NULL)
        register_problem(&(pM->Domain[nl][nd]));
    }
  }
  if (first_call==1) {
    dump_history_enroll(current_z_sq,"<Jz2>");
    dump_history_enroll(div_b_sq,"<divB2>");
    dump_history_enroll(velocity_x_sq,"<vx2>");
    dump_history_enroll(local_eta_value,"<eta>");
    first_call=0;
  }
}
ConsFun_t get_usr_expr(const char *expr)
{
  if (strcmp(expr,"eta")==0) return local_eta_value;
  return NULL;
}
VOutFun_t get_usr_out_fun(const char *name) {(void)name;return NULL;}

#ifdef RESISTIVITY
void get_eta_user(GridS *pG,int i,int j,int k,
                  Real *eta_O,Real *eta_H,Real *eta_A)
{
  *eta_O=local_eta_value(pG,i,j,k);
  *eta_H=0.0;
  *eta_A=0.0;
}
#endif

void Userwork_in_loop(MeshS *pM) {(void)pM;}
void Userwork_after_loop(MeshS *pM) {(void)pM;}
