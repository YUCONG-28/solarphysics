#include "copyright.h"
/*============================================================================*/
/*! \file fluxrope.c
 *  \brief  Problem generator for Flux Rope.
 *
 * PURPOSE: flux rope.
 *
 * REFERENCE: Wang, Shen, & Lin, ApJ 2009 (doi:10.1088/0004-637X/700/2/1716).
 *
 * Update:
 *  2018-03-30
 *    Starting version.
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

/*==============================================================================
 * PRIVATE FUNCTION PROTOTYPES:
 * void flarecs_linetied() - sets BCs on R-x2 boundary
 *============================================================================*/
static void init_azini_from_b(GridS *pGrid);

static void openbc_ox2(GridS *pGrid);
static void linetiedbc_ix2(GridS *pGrid);

/* Functions to compute flux */
static Real func_bphi(const Real r);
static Real func_rjphi(const Real r);
static Real func_uphi(const Real r);

static Real func_bmx(const Real x1, const Real x2);
static Real func_bmy(const Real x1, const Real x2);
static Real func_pbypxini(const Real x1, const Real x2);
static Real func_bphi_over_r_derivative(const Real r);
static Real func_uphi_xy(const Real x, const Real y);

/* Inline adaptive Simpson integration for teaching clarity (single-file view). */
static Real adaptiveSimpsons(Real (*f)(Real), Real a, Real b,
                             Real epsilon, int maxRecursionDepth);
static Real adaptiveSimpsonsAux(Real (*f)(Real), Real a, Real b,
                                Real epsilon, Real S,
                                Real fa, Real fb, Real fc, int bottom);

Real pphi_integrand(Real r);

/* Uniform background gas pressure and density for no-gravity runs */
Real pgas_c, rho_c;

/* Initial conditions */
Real ***pgasini, ***rhoini;
Real ***by;

/* Corner-centered vector potential Az used to recover divergence-free Bx/By. */
Real ***az;

/* Flux parameters */
Real fr_h, fr_hb, fr_ri, fr_del, fr_rja, fr_rmom;
Real fr_xc; // the center of flux rope.

/*=========================== PUBLIC FUNCTIONS ===============================*/
/*----------------------------------------------------------------------------*/
/* problem:  */

void problem(DomainS *pDomain)
{
  /* Step 0: grid handles and local index ranges */
  GridS *pGrid = (pDomain->Grid);
  int i, is = pGrid->is, ie = pGrid->ie;
  int j, js = pGrid->js, je = pGrid->je;
  int k, ks = pGrid->ks, ke = pGrid->ke;
  int n1 = ie - is + 1 + 2 * nghost;
  int n2 = je - js + 1 + 2 * nghost;
  int n3 = ke - ks + 1 + 2 * nghost;

  Real x1, x2, x3;
  Real x1c, x2c, x3c;
  Real ***bx, ***bz;

  /* Step 1: allocate storage for initial thermodynamic and magnetic states */
  /* Initialize pressure and density storage. */
  if ((pgasini = (Real ***)calloc_3d_array(n3, n2, n1, sizeof(Real))) == NULL)
  {
    ath_error("[field_loop]: Error allocating memory for pgasini \n");
  }
  if ((rhoini = (Real ***)calloc_3d_array(n3, n2, n1, sizeof(Real))) == NULL)
  {
    ath_error("[field_loop]: Error allocating memory for rhoini \n");
  }
  /* Corner-centered Az used to construct a discretely divergence-free field. */
  if ((az = (Real ***)calloc_3d_array(n3, n2 + 1, n1 + 1, sizeof(Real))) == NULL)
  {
    ath_error("[field_loop]: Error allocating memory for azini \n");
  }

  /* Step 2: read dimensionless background and flux-rope parameters */
  /* Read input parameters */
  pgas_c = par_getd("problem", "pgas_c");
  rho_c = par_getd("problem", "rho_c");

  /* Initialize flux rope */
  fr_h = par_getd("problem", "h");
  fr_hb = par_getd("problem", "hb");
  fr_ri = par_getd("problem", "ri");
  fr_del = par_getd("problem", "del");
  fr_rmom = par_getd("problem", "rmom");
  fr_rja = par_getd("problem", "rja");

  // Reset fr_h to the cell center.
  k = (int)(fr_h / pGrid->dx2);
  fr_h = (k + 0.5) * pGrid->dx2;
  fr_xc = 0.5 * pGrid->dx1;
  //printf("fr_xc=%.14f, fr_h=%.14f\n", fr_xc, fr_h);

  if ((bx = (Real ***)calloc_3d_array(n3, n2, n1, sizeof(Real))) == NULL)
  {
    ath_error("[field_loop]: Error allocating memory for vector bx\n");
  }
  if ((by = (Real ***)calloc_3d_array(n3, n2, n1, sizeof(Real))) == NULL)
  {
    ath_error("[field_loop]: Error allocating memory for vector by\n");
  }
  if ((bz = (Real ***)calloc_3d_array(n3, n2, n1, sizeof(Real))) == NULL)
  {
    ath_error("[field_loop]: Error allocating memory for vector bz\n");
  }
  /* Step 3: obtain Az from bx, and by. */
  init_azini_from_b(pGrid);

  Real dx = pGrid->dx1, dy = pGrid->dx2;

  /* Step 4: initialize the uniform background and flux-rope pressure offset. */
  Real pgas0 = pgas_c;
  Real rho0 = rho_c;
  for (k = ks; k <= ke; k++)
  {
    for (j = js - nghost; j <= je + nghost; j++)
    {
      for (i = is - nghost; i <= ie + nghost; i++)
      {
        cc_pos(pGrid, i, j, k, &x1c, &x2c, &x3c);
        pgasini[k][j][i] = pgas0 - func_uphi_xy(x1c, x2c);
        rhoini[k][j][i] = rho0 * pow(pgasini[k][j][i] / pgas0,
                                     1.0 / Gamma);
      }
    }
  }

  /* Step 5: evaluate analytic Bx/By directly at their face centers.
  for (k = ks; k <= ke+1; k++)
  {
    for (j = js - nghost; j <= je + nghost; j++)
    {
      for (i = is - nghost; i <= ie + nghost; i++)
      {
        fc_pos(pGrid, i, j, k, &x1, &x2, &x3);
        bx[k][j][i] = func_bmx(x1, x2 + 0.5 * dy);
        by[k][j][i] = func_bmy(x1 + 0.5 * dx, x2);
        bz[k][j][i] = 0.0;
      }
    }
  } */
  /* divergence-free construction from the discrete curl of Az. */
  for (k = ks; k <= ke + 1; k++)
  {
    for (j = js - nghost; j <= je + nghost; j++)
    {
      for (i = is - nghost; i <= ie + nghost; i++)
      {
        bx[k][j][i] = (az[k][j + 1][i] - az[k][j][i]) / dy;
        by[k][j][i] = -(az[k][j][i + 1] - az[k][j][i]) / dx;
        bz[k][j][i] = 0.0;
      }
    }
  }

  /* Step 6: load conserved and face-centered magnetic variables */
  /* All variables */
  for (k = ks; k <= ke; k++)
  {
    for (j = js; j <= je; j++)
    {
      for (i = is; i <= ie; i++)
      {
        /* axis */
        cc_pos(pGrid, i, j, k, &x1, &x2, &x3);

        /* density */
        pGrid->U[k][j][i].d = rhoini[k][j][i];

        /* momentum */
        pGrid->U[k][j][i].M1 = 0.0;
        pGrid->U[k][j][i].M2 = 0.0;
        pGrid->U[k][j][i].M3 = 0.0;

        /* magnetic field */
        pGrid->B1i[k][j][i] = bx[k][j][i];
        pGrid->B2i[k][j][i] = by[k][j][i];
        pGrid->B3i[k][j][i] = bz[k][j][i];

        /* initial magnetic-field values including the boundary and ghost zone */
        if (i == ie && ie > is)
          pGrid->B1i[k][j][i + 1] = bx[k][j][i + 1];
        if (j == je && je > js)
          pGrid->B2i[k][j + 1][i] = by[k][j + 1][i];
        if (k == ke && ke > ks)
          pGrid->B3i[k + 1][j][i] = bz[k + 1][j][i];
      }
    }
  }

  /* Step 7: derive cell-centered magnetic field from face values */
  /* cell-center magnetic field */
  for (k = ks; k <= ke; k++)
  {
    for (j = js; j <= je; j++)
    {
      for (i = is; i <= ie; i++)
      {
        pGrid->U[k][j][i].B1c = 0.5 * (pGrid->B1i[k][j][i] +
                                       pGrid->B1i[k][j][i + 1]);
        pGrid->U[k][j][i].B2c = 0.5 * (pGrid->B2i[k][j][i] +
                                       pGrid->B2i[k][j + 1][i]);
        pGrid->U[k][j][i].B3c = pGrid->B3i[k][j][i];
      }
    }
  }

  /* Step 8: set total energy from gas, magnetic, and kinetic contributions */
  /* total energy */
  for (k = ks; k <= ke; k++)
  {
    for (j = js; j <= je; j++)
    {
      for (i = is; i <= ie; i++)
      {
        pGrid->U[k][j][i].E = pgasini[k][j][i] / (Gamma_1) 
        + 0.5 * (SQR(pGrid->U[k][j][i].B1c) 
               + SQR(pGrid->U[k][j][i].B2c) 
               + SQR(pGrid->U[k][j][i].B3c)) 
        + 0.5 * (SQR(pGrid->U[k][j][i].M1) 
               + SQR(pGrid->U[k][j][i].M2)
               + SQR(pGrid->U[k][j][i].M3)) / pGrid->U[k][j][i].d;
      }
    }
  }

  /* Step 9: optional physics hooks and boundary-condition enrollment */
  /* Set resistivity */
#ifdef RESISTIVITY
  eta_Ohm = par_getd("problem", "eta_Ohm");
  Q_AD = par_getd("problem", "Q_AD");
  Q_Hall = 0.0;
  d_ind = 0.0;
#endif

  /* Set thermal conduction coefficient */

  /* Set viscosity */

  /* Static gravity is intentionally disabled in this no-gravity demo. */

  /* Set optically thin radiative cooling and coronal heating function */

  /* Set boundary value functions */
  /* (a) left-open */

  /* (b) Right-open */

  /* (c) Top-open */
  bvals_mhd_fun(pDomain, right_x2, openbc_ox2);

  /* (d) Bottom-line-tied */
  bvals_mhd_fun(pDomain, left_x2, linetiedbc_ix2);

  /* Step 10: release temporary magnetic work arrays */
  free_3d_array(az);
  free_3d_array(bx);
  /*free_3d_array(by); by[][][] will be used in boundary conditions */
  free_3d_array(bz);
}

/*==============================================================================
 * PROBLEM USER FUNCTIONS:
 * problem_write_restart() - writes problem-specific user data to restart files
 * problem_read_restart()  - reads problem-specific user data from restart files
 * get_usr_expr()          - sets pointer to expression for special output data
 * get_usr_out_fun()       - returns a user defined output function pointer
 * get_usr_par_prop()      - returns a user defined particle selection function
 * Userwork_in_loop        - problem specific work IN     main loop
 * Userwork_after_loop     - problem specific work AFTER  main loop
 *----------------------------------------------------------------------------*/

void problem_write_restart(MeshS *pM, FILE *fp)
{
  return;
}

void problem_read_restart(MeshS *pM, FILE *fp)
{
  return;
}

ConsFun_t get_usr_expr(const char *expr)
{
  return NULL;
}

VOutFun_t get_usr_out_fun(const char *name)
{
  return NULL;
}

#ifdef RESISTIVITY

void get_eta_user(GridS *pG, int i, int j, int k,
                  Real *eta_O, Real *eta_H, Real *eta_A)
{
  return;
}
#endif

void Userwork_in_loop(MeshS *pM)
{
  GridS *pG = pM->Domain[0][0].Grid;
  int is = pG->is, ie = pG->ie;
  int js = pG->js, je = pG->je;
  int ks = pG->ks, ke = pG->ke;
  int i, j, k;
  /*  Pressure floor */
  Real dens_floor = 1.0;
  Real pres_floor = 1.0e-6 * pgas_c;
  Real pres_c;
  Real msqr, bsqr;
  for (k = ks; k <= ke; k++)
  {
    for (j = js; j <= je; j++)
    {
      for (i = is; i <= ie; i++)
      {
        msqr = SQR(pG->U[k][j][i].M1) + SQR(pG->U[k][j][i].M2) + SQR(pG->U[k][j][i].M3);
        bsqr = SQR(pG->U[k][j][i].B1c) + SQR(pG->U[k][j][i].B2c) + SQR(pG->U[k][j][i].B3c);
        pres_c = Gamma_1 * (pG->U[k][j][i].E - 0.5 * msqr / pG->U[k][j][i].d - 0.5 * bsqr);

        pG->U[k][j][i].d = MAX(pG->U[k][j][i].d, dens_floor);
        pres_c = MAX(pres_c, pres_floor);

        pG->U[k][j][i].E = pres_c / Gamma_1 + 0.5 * msqr / pG->U[k][j][i].d + 0.5 * bsqr;
      }
    }
  }
  return;
}

void Userwork_after_loop(MeshS *pM)
{
  return;
}

/*=========================== PRIVATE FUNCTIONS ==============================*/
/*----------------------------------------------------------------------------*/
/*  \fn Initialize corner Az from analytic Bx/By
 *  \brief Build Az on corner points from a reference corner near (fr_xc, fr_h)
 *         and recursively propagate in four directions.
 *
 *  NOTE: This implementation assumes a uniform Cartesian grid in x and y,
 *        i.e., constant dx = pGrid->dx1 and dy = pGrid->dx2 over the domain.
 */
static void init_azini_from_b(GridS *pGrid)
{
  /* Az construction flow:
   *   (1) gather corner coordinates,
   *   (2) choose a reference corner near (fr_xc, fr_h),
   *   (3) propagate Az along x on the reference row,
   *   (4) propagate Az upward/downward, then sweep each row in x. */
  int i, j, k;
  int is = pGrid->is, ie = pGrid->ie;
  int js = pGrid->js, je = pGrid->je;
  int ks = pGrid->ks, ke = pGrid->ke;
  int n1 = ie - is + 1 + 2 * nghost;
  int n2 = je - js + 1 + 2 * nghost;
  int i0 = is - nghost, i1 = ie + nghost + 1;
  int j0 = js - nghost, j1 = je + nghost + 1;
  Real dx = pGrid->dx1, dy = pGrid->dx2;
  Real x1c, x2c, x3c;

  Real *xfc = (Real *)malloc((n1 + 1) * sizeof(Real));
  Real *yfc = (Real *)malloc((n2 + 1) * sizeof(Real));
  if (xfc == NULL || yfc == NULL)
  {
    ath_error("[fluxrope]: Error allocating memory for corner coordinates\n");
  }

  for (k = ks; k <= ke; k++)
  {
    /* (1) x-corner coordinates at this k-plane */
    for (i = i0; i <= i1; i++)
    {
      int ii = i - i0;
      if (i <= ie + nghost)
      {
        cc_pos(pGrid, i, js, k, &x1c, &x2c, &x3c);
        xfc[ii] = x1c - 0.5 * dx;
      }
      else
      {
        cc_pos(pGrid, ie + nghost, js, k, &x1c, &x2c, &x3c);
        xfc[ii] = x1c + 0.5 * dx;
      }
    }

    /* (1) y-corner coordinates at this k-plane */
    for (j = j0; j <= j1; j++)
    {
      int jj = j - j0;
      if (j <= je + nghost)
      {
        cc_pos(pGrid, is, j, k, &x1c, &x2c, &x3c);
        yfc[jj] = x2c - 0.5 * dy;
      }
      else
      {
        cc_pos(pGrid, is, je + nghost, k, &x1c, &x2c, &x3c);
        yfc[jj] = x2c + 0.5 * dy;
      }
    }

    /* (2) choose the corner nearest the model center */
    int ic = 0, jc = 0;
    Real dmin_x = fabs(xfc[0] - fr_xc);
    Real dmin_y = fabs(yfc[0] - fr_h);
    for (i = 1; i <= n1; i++)
    {
      Real dcur_x = fabs(xfc[i] - fr_xc);
      if (dcur_x < dmin_x)
      {
        dmin_x = dcur_x;
        ic = i;
      }
    }
    for (j = 1; j <= n2; j++)
    {
      Real dcur_y = fabs(yfc[j] - fr_h);
      if (dcur_y < dmin_y)
      {
        dmin_y = dcur_y;
        jc = j;
      }
    }

    int iref = i0 + ic;
    int jref = j0 + jc;
    az[k][jref][iref] = 0.0;

    /* (3) propagate on the reference row */
    for (i = iref + 1; i <= i1; i++)
    {
      int ii = i - i0;
      Real xmid = 0.5 * (xfc[ii - 1] + xfc[ii]);
      az[k][jref][i] = az[k][jref][i - 1] - func_bmy(xmid, yfc[jc]) * dx;
    }
    for (i = iref - 1; i >= i0; i--)
    {
      int ii = i - i0;
      Real xmid = 0.5 * (xfc[ii] + xfc[ii + 1]);
      az[k][jref][i] = az[k][jref][i + 1] + func_bmy(xmid, yfc[jc]) * dx;
    }

    /* (4) propagate to rows above, then sweep each row in x */
    for (j = jref + 1; j <= j1; j++)
    {
      int jj = j - j0;
      Real ymid = 0.5 * (yfc[jj - 1] + yfc[jj]);
      az[k][j][iref] = az[k][j - 1][iref] + func_bmx(xfc[ic], ymid) * dy;

      for (i = iref + 1; i <= i1; i++)
      {
        int ii = i - i0;
        Real xmid = 0.5 * (xfc[ii - 1] + xfc[ii]);
        az[k][j][i] = az[k][j][i - 1] - func_bmy(xmid, yfc[jj]) * dx;
      }
      for (i = iref - 1; i >= i0; i--)
      {
        int ii = i - i0;
        Real xmid = 0.5 * (xfc[ii] + xfc[ii + 1]);
        az[k][j][i] = az[k][j][i + 1] + func_bmy(xmid, yfc[jj]) * dx;
      }
    }

    /* (4) propagate to rows below, then sweep each row in x */
    for (j = jref - 1; j >= j0; j--)
    {
      int jj = j - j0;
      Real ymid = 0.5 * (yfc[jj] + yfc[jj + 1]);
      az[k][j][iref] = az[k][j + 1][iref] - func_bmx(xfc[ic], ymid) * dy;

      for (i = iref + 1; i <= i1; i++)
      {
        int ii = i - i0;
        Real xmid = 0.5 * (xfc[ii - 1] + xfc[ii]);
        az[k][j][i] = az[k][j][i - 1] - func_bmy(xmid, yfc[jj]) * dx;
      }
      for (i = iref - 1; i >= i0; i--)
      {
        int ii = i - i0;
        Real xmid = 0.5 * (xfc[ii] + xfc[ii + 1]);
        az[k][j][i] = az[k][j][i + 1] + func_bmy(xmid, yfc[jj]) * dx;
      }
    }
  }

  free(xfc);
  free(yfc);
}

/*----------------------------------------------------------------------------*/
/*! \fn void linetiedbc_ix2(GridS *pGrid)
 *  \brief Sets boundary condition at the bottom.
 */
/*  ix2, line-tied, bottom */
/* Update:
 *   2018-03-06: 
 *   Check reflection at the bottom: m2 = m2 in the ghost zone.*/
void linetiedbc_ix2(GridS *pGrid)
{
  int is = pGrid->is, ie = pGrid->ie;
  int js = pGrid->js;
  int ks = pGrid->ks, ke = pGrid->ke;
  int i, j, k;
  Real p_mj, eb_mj, ek_mj;
  Real b1c_mj, b2c_mj, b3c_mj;
  Real pbypx, x1c, x2c, x3c, x1f, x2f;
  int j_mj;
#ifdef MHD
  int ku;
#endif

  /* Set all variables in ghost zone */
  for (k = ks; k <= ke; k++)
  {
    for (j = 1; j <= nghost; j++)
    {
      for (i = is - nghost; i <= ie + nghost; i++)
      {
        pGrid->U[k][js - j][i] = pGrid->U[k][js][i];
        pGrid->U[k][js - j][i].M1 = 0.0;
        pGrid->U[k][js - j][i].M2 = 0.0;
        pGrid->U[k][js - j][i].M3 = 0.0;
      }
    }
  }

#ifdef MHD
  /* B2i: keep the bottom normal field fixed from precomputed initial by. */
  for (k = ks; k <= ke; k++)
  {
    for (j = 1; j <= nghost; j++)
    {
      for (i = is - nghost; i <= ie + nghost; i++)
      {
        pGrid->B2i[k][js - j][i] = by[k][js - j][i];
      }
    }
  }
  /* Bottom: B1i is not set at i=is-nghost */
  for (k = ks; k <= ke; k++)
  {
    for (j = 1; j <= nghost; j++)
    {
      for (i = is - (nghost - 1); i <= ie + nghost; i++)
      {
        cc_pos(pGrid, i, js-j+1, k, &x1c, &x2c, &x3c);
        x1f = x1c - 0.5 * pGrid->dx1;
        x2f = x2c - 0.5 * pGrid->dx2;
        pbypx = func_pbypxini(x1f, x2f);
        pGrid->B1i[k][js-j][i] = pGrid->B1i[k][js-j+1][i]-(pbypx)*(pGrid->dx2);
      }
    }
  }
  /* Bottom: B3i */
  ku = (pGrid->Nx[2] > 1) ? ke + 1 : ke;
  for (k = ks; k <= ku; k++)
  {
    for (j = 1; j <= nghost; j++)
    {
      for (i = is - nghost; i <= ie + nghost; i++)
      {
        pGrid->B3i[k][js - j][i] = pGrid->B3i[k][js][i];
      }
    }
  }

  /* Reconstruct cell-centered magnetic fields in the bottom ghost cells. */
  for (k = ks; k <= ke; k++)
  {
    for (j = 1; j <= nghost; j++)
    {
      j_mj = js - j;

      for (i = is - (nghost - 1); i <= ie + nghost - 1; i++)
      {
        pGrid->U[k][j_mj][i].B1c =
            0.5 * (pGrid->B1i[k][j_mj][i] +
                   pGrid->B1i[k][j_mj][i + 1]);
      }
      for (i = is - nghost; i <= ie + nghost; i++)
      {
        pGrid->U[k][j_mj][i].B2c =
            0.5 * (pGrid->B2i[k][j_mj][i] +
                   pGrid->B2i[k][j_mj + 1][i]);

        if (pGrid->Nx[2] > 1)
          pGrid->U[k][j_mj][i].B3c =
              0.5 * (pGrid->B3i[k][j_mj][i] +
                     pGrid->B3i[k + 1][j_mj][i]);
        else
          pGrid->U[k][j_mj][i].B3c = pGrid->B3i[k][j_mj][i];
      }
    }
  }

#endif /* MHD */

  /* Pressure and Total Energy */
  for (k = ks; k <= ke; k++)
  {
    for (j = 1; j <= nghost; j++)
    {
      for (i = is - nghost; i <= ie + nghost; i++)
      {
        j_mj = js - j;

        /* bottom boundary: dp/dy = 0 and uniform entropy. */
        ek_mj = 0.5 * (SQR(pGrid->U[k][js][i].M1)
               + SQR(pGrid->U[k][js][i].M2)
               + SQR(pGrid->U[k][js][i].M3)) / pGrid->U[k][js][i].d;
        eb_mj = 0.5 * (SQR(pGrid->U[k][js][i].B1c)
               + SQR(pGrid->U[k][js][i].B2c)
               + SQR(pGrid->U[k][js][i].B3c));
        p_mj = Gamma_1 * (pGrid->U[k][js][i].E - ek_mj - eb_mj);

        eb_mj = 0.5 * (SQR(pGrid->U[k][j_mj][i].B1c)
                     + SQR(pGrid->U[k][j_mj][i].B2c)
                     + SQR(pGrid->U[k][j_mj][i].B3c));
        /* ek_mj = 0 */

        pGrid->U[k][j_mj][i].E = p_mj / Gamma_1 + eb_mj;
      }
    }
  }
  return;
}

/*----------------------------------------------------------------------------*/
/*! \fn static void openbc_ox2(GridS *pGrid)
 *  \brief Open boundary condition at the outer x2 boundary (bc_ox2=2) */
/*----------------------------------------------------------------------------*/
/*  ox2, top boundary */
void openbc_ox2(GridS *pGrid)
{
  int is = pGrid->is, ie = pGrid->ie;
  int je = pGrid->je;
  int ks = pGrid->ks, ke = pGrid->ke;
  int i, j, k;
#ifdef MHD
  int ku; /* k-upper */
#endif

  for (k = ks; k <= ke; k++)
  {
    for (j = 1; j <= nghost; j++)
    {
      for (i = is - nghost; i <= ie + nghost; i++)
      {
        pGrid->U[k][je + j][i] = pGrid->U[k][je - j + 1][i];
        pGrid->U[k][je + j][i].B1c = -pGrid->U[k][je - j + 1][i].B1c;
      }
    }
  }

#ifdef MHD
  /* B1i is not set at i=is-nghost */
  for (k = ks; k <= ke; k++)
  {
    for (j = 1; j <= nghost; j++)
    {
      for (i = is - (nghost - 1); i <= ie + nghost; i++)
      {
        pGrid->B1i[k][je + j][i] = -pGrid->B1i[k][je - j + 1][i];
      }
    }
  }

  /* B2i: j=je+1 is not a boundary condition for the interface field B2i */
  for (k = ks; k <= ke; k++)
  {
    for (j = 2; j <= nghost; j++)
    {
      for (i = is - nghost; i <= ie + nghost; i++)
      {
        pGrid->B2i[k][je + j][i] = pGrid->B2i[k][je - j + 2][i];
      }
    }
  }

  /* B3i: */
  if (pGrid->Nx[2] > 1)
    ku = ke + 1;
  else
    ku = ke;
  for (k = ks; k <= ku; k++)
  {
    for (j = 1; j <= nghost; j++)
    {
      for (i = is - nghost; i <= ie + nghost; i++)
      {
        pGrid->B3i[k][je + j][i] = pGrid->B3i[k][je - j + 1][i];
      }
    }
  }
#endif
  return;
}

/*----------------------------------------------------------------------------*/
/* Adaptive Simpson's rule: integrate f(x) on [a, b] with tolerance epsilon. */
static Real adaptiveSimpsons(Real (*f)(Real), Real a, Real b,
                             Real epsilon, int maxRecursionDepth)
{
  Real c = (a + b) / 2.0;
  Real h = b - a;
  Real fa = f(a), fb = f(b), fc = f(c);
  Real S = (h / 6.0) * (fa + 4.0 * fc + fb);
  return adaptiveSimpsonsAux(f, a, b, epsilon, S, fa, fb, fc,
                             maxRecursionDepth);
}

/* Recursive kernel for adaptiveSimpsons(). */
static Real adaptiveSimpsonsAux(Real (*f)(Real), Real a, Real b,
                                Real epsilon, Real S,
                                Real fa, Real fb, Real fc, int bottom)
{
  Real c = (a + b) / 2.0;
  Real h = b - a;
  Real d = (a + c) / 2.0;
  Real e = (c + b) / 2.0;
  Real fd = f(d), fe = f(e);
  Real Sleft = (h / 12.0) * (fa + 4.0 * fd + fc);
  Real Sright = (h / 12.0) * (fc + 4.0 * fe + fb);
  Real S2 = Sleft + Sright;

  if (bottom <= 0 || fabs(S2 - S) <= 15.0 * epsilon)
    return S2 + (S2 - S) / 15.0;

  return adaptiveSimpsonsAux(f, a, c, epsilon / 2.0, Sleft,
                             fa, fc, fd, bottom - 1) +
         adaptiveSimpsonsAux(f, c, b, epsilon / 2.0, Sright,
                             fc, fb, fe, bottom - 1);
}

/* ============================================================================
 * Functions for flux rope
 * ===========================================================================*/
/*----------------------------------------------------------------------------*/
static Real func_bmx(const Real x1, const Real x2)
{
  /* model field x-component */
  Real rs, rm, rb;
  Real r1, r2;
  Real bmx;

  r1 = fr_ri - 0.5 * fr_del;
  r2 = fr_ri + 0.5 * fr_del;

  rs = sqrt(pow(x1 - fr_xc, 2) + (x2 - fr_h) * (x2 - fr_h));
  rm = sqrt(pow(x1 - fr_xc, 2) + (x2 + fr_h) * (x2 + fr_h));
  rb = sqrt(pow(x1 - fr_xc, 2) + (x2 + fr_hb) * (x2 + fr_hb));

  if (rs > 0.0)
  {
    /* quadrupole */
    //bmx = func_bphi(rs) * (x2 - fr_h) / rs - func_bphi(rm) * (x2 + fr_h) / rm - func_bphi(r2) * fr_rmom * fr_hb * r2 * (x2 + fr_hb) * (3.0 * pow(x1 - fr_xc, 2) - pow((x2 + fr_hb), 2)) / pow(rb, 6);
    /* dipole case */
    bmx = func_bphi(rs) * (x2 - fr_h) / rs - func_bphi(rm) * (x2 + fr_h) / rm - func_bphi(r2) * fr_rmom * fr_hb * r2 * (pow(x1 - fr_xc, 2) - (x2 + fr_hb) * (x2 + fr_hb)) / pow(rb, 4);
  }
  else
  {
    bmx = 0.0;
  }
  return bmx;
}

/*----------------------------------------------------------------------------*/
static Real func_bmy(const Real x1, const Real x2)
{
  /* model field y-component */
  Real rs, rm, rb;
  Real r1, r2;
  Real bmy;

  r1 = fr_ri - 0.5 * fr_del;
  r2 = fr_ri + 0.5 * fr_del;

  rs = sqrt(pow(x1 - fr_xc, 2) + (x2 - fr_h) * (x2 - fr_h));
  rm = sqrt(pow(x1 - fr_xc, 2) + (x2 + fr_h) * (x2 + fr_h));
  rb = sqrt(pow(x1 - fr_xc, 2) + (x2 + fr_hb) * (x2 + fr_hb));

  if (rs > 0.0)
  {
    /* quadrupole case */
    //bmy = -func_bphi(rs) * (x1 - fr_xc) / rs + func_bphi(rm) * (x1 - fr_xc) / rm - func_bphi(r2) * fr_rmom * fr_hb * r2 * (x1 - fr_xc) * (-pow(x1 - fr_xc, 2) + 3.0 * pow((x2 + fr_hb), 2)) / pow(rb, 6);
    /* dipole case */
    bmy = -func_bphi(rs) * (x1 - fr_xc) / rs + func_bphi(rm) * (x1 - fr_xc) / rm - func_bphi(r2) * fr_hb * fr_rmom * r2 * 2.0 * (x1 - fr_xc) * (x2 + fr_hb) / pow(rb, 4);
  }
  else
  {
    bmy = 0.0;
  }
  return bmy;
}

/*----------------------------------------------------------------------------*/
static Real func_pbypxini(const Real x1, const Real x2)
{
  Real x = x1 - fr_xc;
  Real ys = x2 - fr_h;
  Real ym = x2 + fr_h;
  Real yb = x2 + fr_hb;
  Real rs = sqrt(x * x + ys * ys);
  Real rm = sqrt(x * x + ym * ym);
  Real rb = sqrt(x * x + yb * yb);
  Real r2 = fr_ri + 0.5 * fr_del;
  Real dipole_coefficient = func_bphi(r2) * fr_hb * fr_rmom * r2;
  Real pbypx;

  if (rs <= 0.0 || rm <= 0.0 || rb <= 0.0)
    return 0.0;

  pbypx = -func_bphi(rs) / rs
           - x * x * func_bphi_over_r_derivative(rs) / rs
           + func_bphi(rm) / rm
           + x * x * func_bphi_over_r_derivative(rm) / rm
           - 2.0 * dipole_coefficient * yb *
                 (1.0 / pow(rb, 4) - 4.0 * x * x / pow(rb, 6));

  return pbypx;
}

/*----------------------------------------------------------------------------*/
static Real func_bphi_over_r_derivative(const Real r)
{
  Real pi = 3.14159265358979;
  Real r1 = fr_ri - 0.5 * fr_del;
  Real r2 = fr_ri + 0.5 * fr_del;
  Real delq = fr_del * fr_del;
  Real piq = pi * pi;

  if (r <= r1)
  {
    return 0.0;
  }
  else if (r <= r2)
  {
    Real phase = (pi / fr_del) * (r - r1);
    Real profile = 0.5 * r1 * r1 - delq / piq + 0.5 * r * r
                 + (fr_del * r / pi) * sin(phase)
                 + (delq / piq) * cos(phase);
    Real profile_derivative = r * (1.0 + cos(phase));

    return -0.5 * fr_rja *
           (profile_derivative / (r * r) -
            2.0 * profile / (r * r * r));
  }
  else
  {
    Real profile = fr_ri * fr_ri + 0.25 * delq - 2.0 * delq / piq;
    return fr_rja * profile / (r * r * r);
  }
}

/*----------------------------------------------------------------------------*/
static Real func_bphi(const Real r)
{
  /* cylindrical field function */
  Real riq, delq, piq, t1, t2, t3, bphi;
  Real pi = 3.14159265358979;
  Real r1, r2;

  r1 = fr_ri - 0.5 * fr_del;
  r2 = fr_ri + 0.5 * fr_del;

  riq = fr_ri * fr_ri;
  delq = fr_del * fr_del;
  piq = pi * pi;

  if (r <= r1)
  {
    bphi = -0.5 * fr_rja * r;
  }
  else if (r <= r2)
  {
    t1 = 0.5 * r1 * r1 - delq / piq + 0.5 * r * r;
    t2 = (fr_del * r / pi) * sin((pi / fr_del) * (r - r1));
    t3 = (delq / piq) * cos((pi / fr_del) * (r - r1));
    bphi = -0.5 * fr_rja * (t1 + t2 + t3) / r;
  }
  else
  {
    bphi = -0.5 * fr_rja * (riq + 0.25 * delq - 2. * delq / piq) / r;
  }
  return bphi;
}

/*----------------------------------------------------------------------------*/
static Real func_rjphi(const Real r)
{
  /*  current density */
  Real pi = 3.14159265358979;
  Real r1, r2;
  Real rjphi;

  r1 = fr_ri - 0.5 * fr_del;
  r2 = fr_ri + 0.5 * fr_del;

  if (r <= r1)
  {
    rjphi = 1.0 * fr_rja;
  }
  else if (r <= r2)
  {
    rjphi = 0.5 * fr_rja * (cos((pi / fr_del) * (r - r1)) + 1.);
  }
  else
  {
    rjphi = 0.;
  }
  return rjphi;
}

/*----------------------------------------------------------------------------*/
static Real func_uphi_xy(const Real x, const Real y)
{
  Real r = sqrt(pow(x - fr_xc, 2) + pow(y - fr_h, 2));
  return func_uphi(r);
}

/*----------------------------------------------------------------------------*/
static Real func_uphi(const Real r)
{
  Real r2 = fr_ri + 0.5 * fr_del;

  /* pphi_integrand has compact support in r <= r2 through func_rjphi(). */
  if (r >= r2) {
    return 0.0;
  } else {
    return adaptiveSimpsons(pphi_integrand, r, r2, 1.0e-8, 100000);
  }
}

/* ----------------------------------------------------------------------------
 * for integration of pphi
 * ---------------------------------------------------------------------------*/
Real pphi_integrand(Real r)
{
  return func_rjphi(r) * func_bphi(r);
}
