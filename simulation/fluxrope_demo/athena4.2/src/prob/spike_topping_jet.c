#include "copyright.h"
/*============================================================================*/
/*! \file spike_topping_jet.c
 *  \brief Periodic double-Harris current sheets for jet-conditioned Type III
 *         proxy studies.
 *
 * The magnetic field is initialized from a corner-centered vector potential,
 * so the face-centered constrained-transport field is divergence free to
 * roundoff.  The problem is intentionally dimensionless.  Radio emission is
 * added only by the separate Python proxy layer.
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
#error : spike_topping_jet requires MHD
#endif

static Real current_z(const GridS *pG, const int i, const int j, const int k);
static Real div_b(const GridS *pG, const int i, const int j, const int k);
static Real current_z_sq(const GridS *pG, const int i, const int j, const int k);
static Real div_b_sq(const GridS *pG, const int i, const int j, const int k);
static Real velocity_x_sq(
  const GridS *pG, const int i, const int j, const int k
);

static Real flux_function(
  const Real x, const Real y, const Real ly, const Real b0,
  const Real sheet_width, const Real center_fraction,
  const Real perturbation_amplitude, const Real perturbation_width,
  const Real perturbation_kx
)
{
  Real center = center_fraction*ly;
  Real y_lower = -center;
  Real y_upper = center;
  Real equilibrium;
  Real envelope;
  Real perturbation;

  equilibrium = b0*(
    -sheet_width*log(cosh((y-y_lower)/sheet_width))
    +sheet_width*log(cosh((y-y_upper)/sheet_width))
    +y
  );
  envelope = exp(-SQR((y-y_lower)/perturbation_width))
           - exp(-SQR((y-y_upper)/perturbation_width));
  perturbation = perturbation_amplitude*cos(perturbation_kx*x)*envelope;
  return equilibrium + perturbation;
}

void problem(DomainS *pDomain)
{
  GridS *pGrid = pDomain->Grid;
  int i,j,k;
  int is=pGrid->is, ie=pGrid->ie;
  int js=pGrid->js, je=pGrid->je;
  int ks=pGrid->ks, ke=pGrid->ke;
  int nx1=(ie-is)+1+2*nghost;
  int nx2=(je-js)+1+2*nghost;
  int nx3=(ke-ks)+1+2*nghost;
  Real x1c,x2c,x3c,x1f,x2f,x3f;
  Real lx,ly,b0,rho0,beta,sheet_width,center_fraction,guide_field_ratio;
  Real perturbation_amplitude,perturbation_width,perturbation_kx;
  Real p_background,total_pressure,pressure;
  Real ***az;
  static int first_call=1;

  if ((je-js) == 0) {
    ath_error("[spike_topping_jet]: a two-dimensional domain is required\n");
  }

  lx = pDomain->RootMaxX[0] - pDomain->RootMinX[0];
  ly = pDomain->RootMaxX[1] - pDomain->RootMinX[1];
  b0 = par_getd_def("problem","b0",1.0);
  rho0 = par_getd_def("problem","rho0",1.0);
  beta = par_getd_def("problem","beta",1.0);
  sheet_width = par_getd_def("problem","sheet_width",0.20);
  center_fraction = par_getd_def("problem","center_fraction",0.25);
  perturbation_amplitude = par_getd_def("problem","perturbation_amp",0.04);
  perturbation_width = par_getd_def("problem","perturbation_width",0.45);
  perturbation_kx = par_getd_def("problem","perturbation_kx",1.0);
  guide_field_ratio = par_getd_def("problem","guide_field_ratio",0.0);

  if (lx <= 0.0 || ly <= 0.0 || b0 <= 0.0 || rho0 <= 0.0 ||
      beta <= 0.0 || sheet_width <= 0.0 ||
      center_fraction <= 0.0 || center_fraction >= 0.5 ||
      perturbation_width <= 0.0 || guide_field_ratio < 0.0) {
    ath_error("[spike_topping_jet]: invalid positive model parameter\n");
  }

#ifdef RESISTIVITY
  eta_Ohm = par_getd_def("problem","eta_O",0.002);
  Q_Hall = par_getd_def("problem","Q_H",0.0);
  Q_AD = par_getd_def("problem","Q_AD",0.0);
#endif
#ifdef VISCOSITY
  nu_iso = par_getd_def("problem","nu_iso",0.002);
  nu_aniso = par_getd_def("problem","nu_aniso",0.0);
#endif

  if ((az = (Real***)calloc_3d_array(nx3,nx2,nx1,sizeof(Real))) == NULL) {
    ath_error("[spike_topping_jet]: vector-potential allocation failed\n");
  }

  /* Athena uses B1=dAz/dx2 and B2=-dAz/dx1.  Set Az=-psi to match
   * B=(-dpsi/dy,dpsi/dx). */
  for (k=ks; k<=ke+1; k++) {
    for (j=js; j<=je+1; j++) {
      for (i=is; i<=ie+1; i++) {
        cc_pos(pGrid,i,j,k,&x1c,&x2c,&x3c);
        x1f = x1c - 0.5*pGrid->dx1;
        x2f = x2c - 0.5*pGrid->dx2;
        x3f = x3c - 0.5*pGrid->dx3;
        (void)x3f;
        az[k][j][i] = -flux_function(
          x1f,x2f,ly,b0,sheet_width,center_fraction,
          perturbation_amplitude,perturbation_width,perturbation_kx
        );
      }
    }
  }

  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie+1; i++) {
        pGrid->B1i[k][j][i] =
          (az[k][j+1][i]-az[k][j][i])/pGrid->dx2;
      }
    }
  }
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je+1; j++) {
      for (i=is; i<=ie; i++) {
        pGrid->B2i[k][j][i] =
          -(az[k][j][i+1]-az[k][j][i])/pGrid->dx1;
      }
    }
  }
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        pGrid->B3i[k][j][i] = guide_field_ratio*b0;
      }
    }
  }

  p_background = 0.5*beta*b0*b0;
  total_pressure = p_background
    +0.5*b0*b0*(1.0+SQR(guide_field_ratio));
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        pGrid->U[k][j][i].d = rho0;
        pGrid->U[k][j][i].M1 = 0.0;
        pGrid->U[k][j][i].M2 = 0.0;
        pGrid->U[k][j][i].M3 = 0.0;
        pGrid->U[k][j][i].B1c =
          0.5*(pGrid->B1i[k][j][i]+pGrid->B1i[k][j][i+1]);
        pGrid->U[k][j][i].B2c =
          0.5*(pGrid->B2i[k][j][i]+pGrid->B2i[k][j+1][i]);
        pGrid->U[k][j][i].B3c = guide_field_ratio*b0;
        pressure = total_pressure
          - 0.5*(SQR(pGrid->U[k][j][i].B1c)
                 +SQR(pGrid->U[k][j][i].B2c)
                 +SQR(pGrid->U[k][j][i].B3c));
        if (pressure <= 0.0) {
          ath_error("[spike_topping_jet]: pressure balance became non-positive\n");
        }
#ifndef BAROTROPIC
        pGrid->U[k][j][i].E = pressure/Gamma_1
          +0.5*(SQR(pGrid->U[k][j][i].B1c)
                +SQR(pGrid->U[k][j][i].B2c)
                +SQR(pGrid->U[k][j][i].B3c));
#endif
      }
    }
  }

  free_3d_array((void***)az);

  if (first_call == 1) {
    dump_history_enroll(current_z_sq,"<Jz2>");
    dump_history_enroll(div_b_sq,"<divB2>");
    dump_history_enroll(velocity_x_sq,"<vx2>");
    first_call = 0;
  }
}

void problem_write_restart(MeshS *pM, FILE *fp)
{
  (void)pM;
  (void)fp;
  return;
}

void problem_read_restart(MeshS *pM, FILE *fp)
{
  (void)pM;
  (void)fp;
#ifdef RESISTIVITY
  eta_Ohm = par_getd_def("problem","eta_O",0.002);
  Q_Hall = par_getd_def("problem","Q_H",0.0);
  Q_AD = par_getd_def("problem","Q_AD",0.0);
#endif
#ifdef VISCOSITY
  nu_iso = par_getd_def("problem","nu_iso",0.002);
  nu_aniso = par_getd_def("problem","nu_aniso",0.0);
#endif
  return;
}

static Real current_z(const GridS *pG, const int i, const int j, const int k)
{
  return (pG->B2i[k][j][i]-pG->B2i[k][j][i-1])/pG->dx1
       - (pG->B1i[k][j][i]-pG->B1i[k][j-1][i])/pG->dx2;
}

static Real div_b(const GridS *pG, const int i, const int j, const int k)
{
  return (pG->B1i[k][j][i+1]-pG->B1i[k][j][i])/pG->dx1
       + (pG->B2i[k][j+1][i]-pG->B2i[k][j][i])/pG->dx2;
}

static Real current_z_sq(const GridS *pG, const int i, const int j, const int k)
{
  return SQR(current_z(pG,i,j,k));
}

static Real div_b_sq(const GridS *pG, const int i, const int j, const int k)
{
  return SQR(div_b(pG,i,j,k));
}

static Real velocity_x_sq(
  const GridS *pG, const int i, const int j, const int k
)
{
  return SQR(pG->U[k][j][i].M1/pG->U[k][j][i].d);
}

ConsFun_t get_usr_expr(const char *expr)
{
  if (strcmp(expr,"J3") == 0) return current_z;
  if (strcmp(expr,"divB") == 0) return div_b;
  return NULL;
}

VOutFun_t get_usr_out_fun(const char *name)
{
  (void)name;
  return NULL;
}

#ifdef RESISTIVITY
void get_eta_user(
  GridS *pG, int i, int j, int k,
  Real *eta_O, Real *eta_H, Real *eta_A
)
{
  (void)pG;
  (void)i;
  (void)j;
  (void)k;
  *eta_O = eta_Ohm;
  *eta_H = 0.0;
  *eta_A = 0.0;
}
#endif

void Userwork_in_loop(MeshS *pM)
{
  (void)pM;
}

void Userwork_after_loop(MeshS *pM)
{
  (void)pM;
}
