module mod_usr
  use mod_mhd
  implicit none
  double precision :: q_e, parb,unit_currentdensity

contains

  subroutine usr_init()
    call set_coordinate_system("Cartesian_2.5D")

    unit_length        = 1.d9 ! cm
    unit_temperature   = 1.d6 ! K
    unit_numberdensity = 1.d9 ! cm-3,cm-3

    usr_init_one_grid       => initonegrid_usr
    usr_special_bc          => specialbound_usr
    usr_aux_output          => specialvar_output
    usr_add_aux_names       => specialvarnames_output
    usr_set_B0              => specialset_B0
    usr_set_J0              => specialset_J0
    usr_special_convert     => usrspecial_convert
    usr_special_resistivity => special_eta
    usr_var_for_errest      => p_for_errest

    call mhd_activate()
    parb=20.d0/3.d0
    ! unit of current density
    unit_currentdensity=unit_magneticfield/unit_length/4.d0/dpi
    ! unit of charge
    q_e=unit_currentdensity/unit_numberdensity/unit_velocity
    if(mype==0) print*,'unit of charge',q_e
    ! dimensionless charge of electron
    q_e=1.60217653d-19/q_e
    if(mype==0) print*,'dimensionless e',q_e

  end subroutine usr_init

  subroutine initonegrid_usr(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
     ixOmax1,ixOmax2,w,x)
  ! initialize one grid
    integer, intent(in) :: ixImin1,ixImin2,ixImax1,ixImax2, ixOmin1,ixOmin2,&
       ixOmax1,ixOmax2
    double precision, intent(in) :: x(ixImin1:ixImax1,ixImin2:ixImax2,1:ndim)
    double precision, intent(inout) :: w(ixImin1:ixImax1,ixImin2:ixImax2,1:nw)
    double precision :: Bf(ixImin1:ixImax1,ixImin2:ixImax2,1:ndir)
    double precision :: htra, wtra, rpho, parb
    logical, save:: first=.true.

    if (first) then
       if (mype==0) then
          print *,'YOKOYAMA and SHIBATA 2001 ApJ'
       end if
       first=.false.
    end if
    rpho=1.d5 ! number density at the bottom relaxla
    htra=0.3d0 ! height of initial transition region
    wtra=0.06d0 ! width of initial transition region
    w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,rho_)=1.d0+&
       (rpho-1.d0)*(1.d0-tanh((x(ixOmin1:ixOmax1,ixOmin2:ixOmax2,&
       2)-htra)/wtra))/2.d0
    w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,p_)=1.d0
    w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,mom(:))=zero
    if(B0field) then
      w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,mag(:))=zero
    else
      call specialset_B0(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
         ixOmax1,ixOmax2,x,Bf)
      w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,mag(1:ndir))=Bf(ixOmin1:ixOmax1,&
         ixOmin2:ixOmax2,1:ndir)
    end if
    if(mhd_glm) w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,psi_)=0.d0
    call mhd_to_conserved(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
       ixOmax1,ixOmax2,w,x)
  end subroutine initonegrid_usr

  subroutine specialbound_usr(qt,ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,&
     ixOmin2,ixOmax1,ixOmax2,iB,w,x)
    ! special boundary types, user defined
    integer, intent(in) :: ixOmin1,ixOmin2,ixOmax1,ixOmax2, iB, ixImin1,&
       ixImin2,ixImax1,ixImax2
    double precision, intent(in) :: qt, x(ixImin1:ixImax1,ixImin2:ixImax2,&
       1:ndim)
    double precision, intent(inout) :: w(ixImin1:ixImax1,ixImin2:ixImax2,1:nw)
    double precision :: pth(ixImin1:ixImax1,ixImin2:ixImax2)
    integer :: ix1,ix2, ixAmin1,ixAmin2,ixAmax1,ixAmax2

    select case(iB)
    case(1)
      ixAmin1=ixOmin1;ixAmin2=ixOmin2;ixAmax1=ixOmax1;ixAmax2=ixOmax2;
      ixAmin1=ixOmax1+1;ixAmax1=ixOmax1+nghostcells;
      call mhd_get_pthermal(w,x,ixImin1,ixImin2,ixImax1,ixImax2,ixAmin1,&
         ixAmin2,ixAmax1,ixAmax2,pth)
      !w(ixO^S,rho_)=w(ixOmax1+nghostcells:ixOmax1+1:-1,ixOmin2:ixOmax2,rho_)
      !w(ixO^S,p_)=pth(ixOmax1+nghostcells:ixOmax1+1:-1,ixOmin2:ixOmax2)
      do ix1=ixOmin1,ixOmax1
        w(ix1,ixOmin2:ixOmax2,mom(1))=w(ixOmax1+1,ixOmin2:ixOmax2,&
           mom(1))/w(ixOmax1+1,ixOmin2:ixOmax2,rho_)
        w(ix1,ixOmin2:ixOmax2,mom(2))=w(ixOmax1+1,ixOmin2:ixOmax2,&
           mom(2))/w(ixOmax1+1,ixOmin2:ixOmax2,rho_)
        w(ix1,ixOmin2:ixOmax2,mom(3))=w(ixOmax1+1,ixOmin2:ixOmax2,&
           mom(3))/w(ixOmax1+1,ixOmin2:ixOmax2,rho_)
        w(ix1,ixOmin2:ixOmax2,rho_)=w(ixOmax1+1,ixOmin2:ixOmax2,rho_)
        w(ix1,ixOmin2:ixOmax2,p_)=pth(ixOmax1+1,ixOmin2:ixOmax2)
      enddo
      do ix1=ixOmax1,ixOmin1,-1
        w(ix1,ixOmin2:ixOmax2,mag(:))=(1.0d0/3.0d0)* (-w(ix1+2,ixOmin2:ixOmax2,&
           mag(:)) +4.0d0*w(ix1+1,ixOmin2:ixOmax2,mag(:)))
      !  w(ix1,ixOmin2:ixOmax2,mom(:))=(1.0d0/3.0d0)* &
      !             (-w(ix1+2,ixOmin2:ixOmax2,mom(:)) &
      !        +4.0d0*w(ix1+1,ixOmin2:ixOmax2,mom(:)))
      enddo
      call mhd_to_conserved(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
         ixOmax1,ixOmax2,w,x)
    case(2)
      ixAmin1=ixOmin1;ixAmin2=ixOmin2;ixAmax1=ixOmax1;ixAmax2=ixOmax2;
      ixAmin1=ixOmin1-nghostcells;ixAmax1=ixOmin1-1;
      call mhd_get_pthermal(w,x,ixImin1,ixImin2,ixImax1,ixImax2,ixAmin1,&
         ixAmin2,ixAmax1,ixAmax2,pth)
      !w(ixO^S,rho_)=w(ixOmin1-1:ixOmin1-nghostcells:-1,ixOmin2:ixOmax2,rho_)
      !w(ixO^S,p_)=pth(ixOmin1-1:ixOmin1-nghostcells:-1,ixOmin2:ixOmax2)
      do ix1=ixOmin1,ixOmax1
        w(ix1,ixOmin2:ixOmax2,mom(1))=w(ixOmin1-1,ixOmin2:ixOmax2,&
           mom(1))/w(ixOmin1-1,ixOmin2:ixOmax2,rho_)
        w(ix1,ixOmin2:ixOmax2,mom(2))=w(ixOmin1-1,ixOmin2:ixOmax2,&
           mom(2))/w(ixOmin1-1,ixOmin2:ixOmax2,rho_)
        w(ix1,ixOmin2:ixOmax2,mom(3))=w(ixOmin1-1,ixOmin2:ixOmax2,&
           mom(3))/w(ixOmin1-1,ixOmin2:ixOmax2,rho_)
        w(ix1,ixOmin2:ixOmax2,rho_)=w(ixOmin1-1,ixOmin2:ixOmax2,rho_)
        w(ix1,ixOmin2:ixOmax2,p_)=pth(ixOmin1-1,ixOmin2:ixOmax2)
      enddo
      do ix1=ixOmin1,ixOmax1
        w(ix1,ixOmin2:ixOmax2,mag(:))=(1.0d0/3.0d0)* (-w(ix1-2,ixOmin2:ixOmax2,&
           mag(:)) +4.0d0*w(ix1-1,ixOmin2:ixOmax2,mag(:)))
        !w(ix1,ixOmin2:ixOmax2,mom(:))=(1.0d0/3.0d0)* &
        !           (-w(ix1-2,ixOmin2:ixOmax2,mom(:)) &
        !      +4.0d0*w(ix1-1,ixOmin2:ixOmax2,mom(:)))
      enddo
      call mhd_to_conserved(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
         ixOmax1,ixOmax2,w,x)
    case(3)
      ixAmin1=ixOmin1;ixAmin2=ixOmin2;ixAmax1=ixOmax1;ixAmax2=ixOmax2;
      ixAmin2=ixOmax2+1;ixAmax2=ixOmax2+1;
      call mhd_get_pthermal(w,x,ixImin1,ixImin2,ixImax1,ixImax2,ixAmin1,&
         ixAmin2,ixAmax1,ixAmax2,pth)
      do ix2=ixOmin2,ixOmax2
        w(ixOmin1:ixOmax1,ix2,rho_)=w(ixOmin1:ixOmax1,ixOmax2+1,rho_)
        w(ixOmin1:ixOmax1,ix2,mom(1))=w(ixOmin1:ixOmax1,ixOmax2+1,&
           mom(1))/w(ixOmin1:ixOmax1,ixOmax2+1,rho_)
        w(ixOmin1:ixOmax1,ix2,mag(2))=w(ixOmin1:ixOmax1,ixOmax2+1,mag(2))
        w(ixOmin1:ixOmax1,ix2,mag(3))=w(ixOmin1:ixOmax1,ixOmax2+1,mag(3))
        w(ixOmin1:ixOmax1,ix2,p_)=pth(ixOmin1:ixOmax1,ixOmax2+1)
        if(mhd_glm) w(ixOmin1:ixOmax1,ix2,psi_)=w(ixOmin1:ixOmax1,ixOmax2+1,&
           psi_)
      enddo
      w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,mom(2:3))=zero
      w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,mag(1))=zero
      call mhd_to_conserved(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
         ixOmax1,ixOmax2,w,x)
    case(4)
      ixAmin1=ixOmin1;ixAmin2=ixOmin2;ixAmax1=ixOmax1;ixAmax2=ixOmax2;
      ixAmin2=ixOmin2-1;ixAmax2=ixOmin2-1;
      call mhd_get_pthermal(w,x,ixImin1,ixImin2,ixImax1,ixImax2,ixAmin1,&
         ixAmin2,ixAmax1,ixAmax2,pth)
      do ix2=ixOmin2,ixOmax2
        w(ixOmin1:ixOmax1,ix2,rho_)=w(ixOmin1:ixOmax1,ixOmin2-1,rho_)
        w(ixOmin1:ixOmax1,ix2,mom(1))=w(ixOmin1:ixOmax1,ixOmin2-1,&
           mom(1))/w(ixOmin1:ixOmax1,ixOmin2-1,rho_)
        w(ixOmin1:ixOmax1,ix2,mom(2))=w(ixOmin1:ixOmax1,ixOmin2-1,&
           mom(2))/w(ixOmin1:ixOmax1,ixOmin2-1,rho_)
        w(ixOmin1:ixOmax1,ix2,mom(3))=w(ixOmin1:ixOmax1,ixOmin2-1,&
           mom(3))/w(ixOmin1:ixOmax1,ixOmin2-1,rho_)
        w(ixOmin1:ixOmax1,ix2,p_)=pth(ixOmin1:ixOmax1,ixOmin2-1)
      enddo
      do ix2=ixOmin2,ixOmax2
        w(ixOmin1:ixOmax1,ix2,mag(:))=(1.0d0/3.0d0)* (-w(ixOmin1:ixOmax1,ix2-2,&
           mag(:))+4.0d0*w(ixOmin1:ixOmax1,ix2-1,mag(:)))
      enddo
      call mhd_to_conserved(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
         ixOmax1,ixOmax2,w,x)
    case default
      call mpistop("Special boundary is not defined for this region")
    end select
  end subroutine specialbound_usr

  subroutine p_for_errest(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
     ixOmax1,ixOmax2,iflag,w,x,var)
    integer, intent(in)           :: ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,&
       ixOmin2,ixOmax1,ixOmax2,iflag
    double precision, intent(in)  :: w(ixImin1:ixImax1,ixImin2:ixImax2,1:nw),&
       x(ixImin1:ixImax1,ixImin2:ixImax2,1:ndim)
    double precision, intent(out) :: var(ixImin1:ixImax1,ixImin2:ixImax2)

    call mhd_get_pthermal(w,x,ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
       ixOmax1,ixOmax2,var)

  end subroutine p_for_errest

  subroutine specialvar_output(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
     ixOmax1,ixOmax2,w,x,normconv)
  ! this subroutine can be used in convert, to add auxiliary variables to the
  ! converted output file, for further analysis using tecplot, paraview, ....
  ! these auxiliary values need to be stored in the nw+1:nw+nwauxio slots
  !
  ! the array normconv can be filled in the (nw+1:nw+nwauxio) range with
  ! corresponding normalization values (default value 1)
    integer, intent(in)                :: ixImin1,ixImin2,ixImax1,ixImax2,&
       ixOmin1,ixOmin2,ixOmax1,ixOmax2
    double precision, intent(in)       :: x(ixImin1:ixImax1,ixImin2:ixImax2,&
       1:ndim)
    double precision                   :: w(ixImin1:ixImax1,ixImin2:ixImax2,&
       nw+nwauxio)
    double precision                   :: normconv(0:nw+nwauxio)
    double precision :: pth(ixImin1:ixImax1,ixImin2:ixImax2),&
       B2(ixImin1:ixImax1,ixImin2:ixImax2),divb(ixImin1:ixImax1,&
       ixImin2:ixImax2)
    double precision :: Btotal(ixImin1:ixImax1,ixImin2:ixImax2,1:ndir),&
       current_o(ixImin1:ixImax1,ixImin2:ixImax2,3)
    integer :: idir,idirmin

    ! output temperature
    call mhd_get_pthermal(w,x,ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
       ixOmax1,ixOmax2,pth)
    w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,nw+1)=pth(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2)/w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,rho_)
    if(B0field) then
      Btotal(ixImin1:ixImax1,ixImin2:ixImax2,1:ndir)=w(ixImin1:ixImax1,&
         ixImin2:ixImax2,mag(1:ndir))+block%B0(ixImin1:ixImax1,ixImin2:ixImax2,&
         1:ndir,0)
    else
      Btotal(ixImin1:ixImax1,ixImin2:ixImax2,1:ndir)=w(ixImin1:ixImax1,&
         ixImin2:ixImax2,mag(1:ndir))
    endif
    ! B^2
    B2(ixOmin1:ixOmax1,ixOmin2:ixOmax2)=sum((Btotal(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2,:))**2,dim=ndim+1)
    ! output Alfven wave speed B/sqrt(rho)
    w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,nw+2)=dsqrt(B2(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2)/w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,rho_))
    ! output divB1
    call divvector(Btotal,ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
       ixOmax1,ixOmax2,divb)
    w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,nw+3)=0.5d0*divb(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2)/dsqrt(B2(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2))/(1.0d0/dxlevel(1)+1.0d0/dxlevel(2))
    ! output the plasma beta p*2/B**2
    w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,nw+4)=pth(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2)*two/B2(ixOmin1:ixOmax1,ixOmin2:ixOmax2)
    ! output current
    call get_current(w,ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,ixOmax1,&
       ixOmax2,idirmin,current_o)
    w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,nw+5)=current_o(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2,1)
    w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,nw+6)=current_o(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2,2)
    w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,nw+7)=current_o(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2,3)
    ! output special resistivity eta
    call special_eta(w,ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,ixOmax1,&
       ixOmax2,idirmin,x,current_o,divb)
    w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,nw+8)=divb(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2)

  end subroutine specialvar_output

  subroutine specialvarnames_output(varnames)
  ! newly added variables need to be concatenated with the w_names/primnames string
    character(len=*) :: varnames
    varnames='Te Alfv divB beta j1 j2 j3 eta'
  end subroutine specialvarnames_output

  subroutine specialset_B0(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
     ixOmax1,ixOmax2,x,wB0)
  ! Here add a time-independent background magnetic field
    integer, intent(in)           :: ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,&
       ixOmin2,ixOmax1,ixOmax2
    double precision, intent(in)  :: x(ixImin1:ixImax1,ixImin2:ixImax2,1:ndim)
    double precision, intent(inout) :: wB0(ixImin1:ixImax1,ixImin2:ixImax2,&
       1:ndir)

    wB0(ixOmin1:ixOmax1,ixOmin2:ixOmax2,1)=zero
    wB0(ixOmin1:ixOmax1,ixOmin2:ixOmax2,2)=-Busr*dtanh(parb*x(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2,1))
    wB0(ixOmin1:ixOmax1,ixOmin2:ixOmax2,3)=Busr/dcosh(parb*x(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2,1))

  end subroutine specialset_B0

  subroutine specialset_J0(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
     ixOmax1,ixOmax2,x,wJ0)
  ! Here add a time-independent background current density
    integer, intent(in)           :: ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,&
       ixOmin2,ixOmax1,ixOmax2
    double precision, intent(in)  :: x(ixImin1:ixImax1,ixImin2:ixImax2,1:ndim)
    double precision, intent(inout) :: wJ0(ixImin1:ixImax1,ixImin2:ixImax2,&
       7-2*ndir:ndir)

    wJ0(ixOmin1:ixOmax1,ixOmin2:ixOmax2,1)=zero
    wJ0(ixOmin1:ixOmax1,ixOmin2:ixOmax2,2)=parb*Busr*dtanh(parb*x(&
       ixOmin1:ixOmax1,ixOmin2:ixOmax2,1))/dcosh(parb*x(ixOmin1:ixOmax1,&
       ixOmin2:ixOmax2,1))
    wJ0(ixOmin1:ixOmax1,ixOmin2:ixOmax2,3)=-&
       parb*Busr/dcosh(parb*x(ixOmin1:ixOmax1,ixOmin2:ixOmax2,1))**2

  end subroutine specialset_J0

  subroutine special_eta(w,ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
     ixOmax1,ixOmax2,idirmin,x,current,eta)
    ! Set the common "eta" array for resistive MHD based on w or the
    ! "current" variable which has components between idirmin and 3.
    integer, intent(in) :: ixImin1,ixImin2,ixImax1,ixImax2, ixOmin1,ixOmin2,&
       ixOmax1,ixOmax2, idirmin
    double precision, intent(in) :: w(ixImin1:ixImax1,ixImin2:ixImax2,nw),&
        x(ixImin1:ixImax1,ixImin2:ixImax2,1:ndim)
    double precision :: current(ixImin1:ixImax1,ixImin2:ixImax2,7-2*ndir:3),&
        eta(ixImin1:ixImax1,ixImin2:ixImax2)
    double precision :: rad(ixImin1:ixImax1,ixImin2:ixImax2),heta,reta,eta1,&
       eta2,etam,vc,tar
    double precision :: jc,jabs(ixImin1:ixImax1,ixImin2:ixImax2)

    heta = 6.
    reta = 0.8d0 * 0.3d0
    eta1 = 0.002d0
    tar= 0.4d0
    !tar= 0.0d0
    vc=1.d-4
    eta2=4.d-3
    etam=4.d-1
    if (global_time<tar) then
      rad(ixOmin1:ixOmax1,ixOmin2:ixOmax2)=dsqrt(x(ixOmin1:ixOmax1,&
         ixOmin2:ixOmax2,1)**2+(x(ixOmin1:ixOmax1,ixOmin2:ixOmax2,2)-heta)**2)
      where (rad(ixOmin1:ixOmax1,ixOmin2:ixOmax2) .lt. reta)
        eta(ixOmin1:ixOmax1,ixOmin2:ixOmax2)=eta1*(2.d0*(rad(ixOmin1:ixOmax1,&
           ixOmin2:ixOmax2)/reta)**3-3.d0*(rad(ixOmin1:ixOmax1,&
           ixOmin2:ixOmax2)/reta)**2+1.d0)
      elsewhere
        eta(ixOmin1:ixOmax1,ixOmin2:ixOmax2)=zero
      endwhere
    else
      rad(ixOmin1:ixOmax1,ixOmin2:ixOmax2)=dsqrt(sum(current(ixOmin1:ixOmax1,&
         ixOmin2:ixOmax2,:)**2,dim=ndim+1))/w(ixOmin1:ixOmax1,ixOmin2:ixOmax2,&
         rho_)/q_e
      where(rad(ixOmin1:ixOmax1,ixOmin2:ixOmax2)>vc)
        eta(ixOmin1:ixOmax1,ixOmin2:ixOmax2)=eta2*(rad(ixOmin1:ixOmax1,&
           ixOmin2:ixOmax2)/vc-1.d0)
      elsewhere
        eta(ixOmin1:ixOmax1,ixOmin2:ixOmax2)=0.d0
      endwhere
      where(eta(ixOmin1:ixOmax1,ixOmin2:ixOmax2)>etam)
        eta(ixOmin1:ixOmax1,ixOmin2:ixOmax2)=etam
      endwhere
    end if

    end subroutine special_eta

  subroutine usrspecial_convert(qunitconvert)
    integer, intent(in) :: qunitconvert
    character(len=20):: userconvert_type

    call spatial_integral_w
  end subroutine usrspecial_convert

  subroutine spatial_integral_w
    double precision :: dvolume(ixGlo1:ixGhi1,ixGlo2:ixGhi2),&
        dsurface(ixGlo1:ixGhi1,ixGlo2:ixGhi2),timephy,dvone
    double precision, allocatable :: integral_ipe(:), integral_w(:)

    integer           :: nregions,ireg,ncellpe,ncell,idims,hxMlo1,hxMlo2,&
       hxMhi1,hxMhi2,nx1,nx2
    integer           :: iigrid,igrid,status(MPI_STATUS_SIZE),ni
    character(len=100):: filename,region
    character(len=1024) :: line, datastr
    logical           :: patchwi(ixGlo1:ixGhi1,ixGlo2:ixGhi2),alive

    nregions=1
    ! number of integrals to perform
    ni=3
    allocate(integral_ipe(ni),integral_w(ni))
    integral_ipe=0.d0
    integral_w=0.d0
    nx1=ixMhi1-ixMlo1+1;nx2=ixMhi2-ixMlo2+1;
    do ireg=1,nregions
      select case(ireg)
      case(1)
        region='fulldomain'
      case(2)
        region='cropped'
      end select
      ncellpe=0
      do iigrid=1,igridstail; igrid=igrids(iigrid);
        block=>ps(igrid)
        if(slab) then
          dvone=rnode(rpdx1_,igrid)*rnode(rpdx2_,igrid)
          dvolume(ixMlo1:ixMhi1,ixMlo2:ixMhi2)=dvone
          dsurface(ixMlo1:ixMhi1,ixMlo2:ixMhi2)=two*(dvone/rnode(rpdx1_,&
             igrid)+dvone/rnode(rpdx2_,igrid))
        else
          dvolume(ixMlo1:ixMhi1,ixMlo2:ixMhi2)=ps(igrid)%dvolume(ixMlo1:ixMhi1,&
             ixMlo2:ixMhi2)
          dsurface(ixMlo1:ixMhi1,ixMlo2:ixMhi2)= &
             sum(ps(igrid)%surfaceC(ixMlo1:ixMhi1,ixMlo2:ixMhi2,:),dim=ndim+1)
          do idims=1,ndim
            hxMlo1=ixMlo1-kr(idims,1);hxMlo2=ixMlo2-kr(idims,2)
            hxMhi1=ixMhi1-kr(idims,1);hxMhi2=ixMhi2-kr(idims,2);
            dsurface(ixMlo1:ixMhi1,ixMlo2:ixMhi2)=dsurface(ixMlo1:ixMhi1,&
               ixMlo2:ixMhi2)+ps(igrid)%surfaceC(hxMlo1:hxMhi1,hxMlo2:hxMhi2,&
               idims)
          end do
        end if
        dxlevel(1)=rnode(rpdx1_,igrid);dxlevel(2)=rnode(rpdx2_,igrid);
        patchwi(ixGlo1:ixGhi1,ixGlo2:ixGhi2)=.false.
        select case(region)
        case('cropped')
           call mask_grid(ixGlo1,ixGlo2,ixGhi1,ixGhi2,ixMlo1,ixMlo2,ixMhi1,&
              ixMhi2,ps(igrid)%w,ps(igrid)%x,patchwi,ncellpe)
        case('fulldomain')
           patchwi(ixMlo1:ixMhi1,ixMlo2:ixMhi2)=.true.
           ncellpe=ncellpe+nx1*nx2
        case default
           call mpistop("region not defined")
        end select
        integral_ipe(1)=integral_ipe(1)+ integral_grid(ixGlo1,ixGlo2,ixGhi1,&
           ixGhi2,ixMlo1,ixMlo2,ixMhi1,ixMhi2,ps(igrid)%w,ps(igrid)%x,dvolume,&
           dsurface,1,patchwi)
        integral_ipe(2)=integral_ipe(2)+ integral_grid(ixGlo1,ixGlo2,ixGhi1,&
           ixGhi2,ixMlo1,ixMlo2,ixMhi1,ixMhi2,ps(igrid)%w,ps(igrid)%x,dvolume,&
           dsurface,2,patchwi)
        integral_ipe(3)=integral_ipe(3)+ integral_grid(ixGlo1,ixGlo2,ixGhi1,&
           ixGhi2,ixMlo1,ixMlo2,ixMhi1,ixMhi2,ps(igrid)%w,ps(igrid)%x,dvolume,&
           dsurface,3,patchwi)
      end do
      call MPI_ALLREDUCE(integral_ipe,integral_w,ni,MPI_DOUBLE_PRECISION,&
         MPI_SUM,icomm,ierrmpi)
      !call MPI_ALLREDUCE(ncellpe,ncell,1,MPI_INTEGER,MPI_SUM,icomm,ierrmpi)
      timephy=global_time
      if(mype==0) then
        write(filename,"(a,a,a)") TRIM(base_filename),TRIM(region),"mkc.csv"
        inquire(file=filename,exist=alive)
        if(alive) then
          open(unit=21,file=filename,form='formatted',status='old',&
             access='append')
        else
          open(unit=21,file=filename,form='formatted',status='new')
          write(21,'(a)') 'time, emagnetic, einternal, current'
        endif
        write(datastr,'(es13.6, a)') timephy,','
        line=datastr
        write(datastr,"(es13.6, a)") integral_w(1),','
        line = trim(line)//trim(datastr)
        write(datastr,"(es13.6, a)") integral_w(2),','
        line = trim(line)//trim(datastr)
        write(datastr,"(es13.6)") integral_w(3)
        line = trim(line)//trim(datastr)
        write(21,'(a)') trim(line)
        close(21)
      endif
    enddo
    deallocate(integral_ipe,integral_w)
  end subroutine spatial_integral_w

  subroutine mask_grid(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,ixOmax1,&
     ixOmax2,w,x,patchwi,cellcount)
    integer, intent(in)                :: ixImin1,ixImin2,ixImax1,ixImax2,&
       ixOmin1,ixOmin2,ixOmax1,ixOmax2
    double precision, intent(in)       :: x(ixImin1:ixImax1,ixImin2:ixImax2,&
       1:ndim)
    double precision                   :: w(ixImin1:ixImax1,ixImin2:ixImax2,&
       nw+nwauxio)
    logical, intent(inout)             :: patchwi(ixGlo1:ixGhi1,ixGlo2:ixGhi2)

    double precision  ::  buff
    integer                            :: ix1,ix2,cellcount

    buff=0.05d0*(xprobmax1-xprobmin1)
    do ix2=ixOmin2,ixOmax2
    do ix1=ixOmin1,ixOmax1
       if(x(ix1,ix2,1)>xprobmin1+buff .and. x(ix1,ix2,&
          1)<xprobmax1-buff .and. x(ix1,ix2,2)>xprobmin2+buff .and. x(ix1,ix2,&
          2)<xprobmax2-buff) then
         patchwi(ix1,ix2)=.true.
         cellcount=cellcount+1
       else
         patchwi(ix1,ix2)=.false.
       endif
    end do
    end do
    return

  end subroutine mask_grid

  function integral_grid(ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
     ixOmax1,ixOmax2,w,x,dvolume,dsurface,intval,patchwi)
    integer, intent(in)                :: ixImin1,ixImin2,ixImax1,ixImax2,&
       ixOmin1,ixOmin2,ixOmax1,ixOmax2,intval
    double precision, intent(in)       :: x(ixImin1:ixImax1,ixImin2:ixImax2,&
       1:ndim),dvolume(ixGlo1:ixGhi1,ixGlo2:ixGhi2),dsurface(ixGlo1:ixGhi1,&
       ixGlo2:ixGhi2)
    double precision, intent(in)       :: w(ixImin1:ixImax1,ixImin2:ixImax2,&
       nw)
    logical, intent(in) :: patchwi(ixGlo1:ixGhi1,ixGlo2:ixGhi2)

    double precision, dimension(ixGlo1:ixGhi1,ixGlo2:ixGhi2,1:ndir) :: bvec,&
       qvec
    double precision :: current(ixGlo1:ixGhi1,ixGlo2:ixGhi2,7-2*ndir:3),&
       tmp(ixGlo1:ixGhi1,ixGlo2:ixGhi2)
    double precision :: integral_grid,mcurrent
    integer :: ix1,ix2,idirmin,idir,jdir,kdir

    integral_grid=0.d0
    select case(intval)
     case(1)
      ! magnetic energy
      if(B0field)then
        tmp(ixOmin1:ixOmax1,ixOmin2:ixOmax2)=0.5d0*sum((w(ixOmin1:ixOmax1,&
           ixOmin2:ixOmax2,mag(:))+block%B0(ixOmin1:ixOmax1,ixOmin2:ixOmax2,:,&
           0))**2,dim=ndim+1)
      else
        tmp(ixOmin1:ixOmax1,ixOmin2:ixOmax2)=0.5d0*sum(w(ixOmin1:ixOmax1,&
           ixOmin2:ixOmax2,mag(:))**2,dim=ndim+1)
      endif
      do ix2=ixOmin2,ixOmax2
      do ix1=ixOmin1,ixOmax1
         if(patchwi(ix1,ix2)) integral_grid=integral_grid+tmp(ix1,&
            ix2)*dvolume(ix1,ix2)
      end do
      end do
     case(2)
      ! internal energy
      call mhd_get_pthermal(w,x,ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,&
         ixOmin2,ixOmax1,ixOmax2,tmp)
      do ix2=ixOmin2,ixOmax2
      do ix1=ixOmin1,ixOmax1
         if(patchwi(ix1,ix2))  integral_grid=integral_grid+tmp(ix1,&
            ix2)/(mhd_gamma-1.d0)*dvolume(ix1,ix2)
      end do
      end do
     case(3)
      ! current strength
      call get_current(w,ixImin1,ixImin2,ixImax1,ixImax2,ixOmin1,ixOmin2,&
         ixOmax1,ixOmax2,idirmin,current)
      do ix2=ixOmin2,ixOmax2
      do ix1=ixOmin1,ixOmax1
         if(patchwi(ix1,ix2)) integral_grid=integral_grid+&
            dsqrt(sum(current(ix1,ix2,:)**2))*dvolume(ix1,ix2)
      end do
      end do
     case default
         call mpistop("intval not defined")
    end select

    return
  end function integral_grid

end module mod_usr
