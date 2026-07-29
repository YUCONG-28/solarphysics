! Development case for the shared Spike-Topping 2.5D solar-jet initial state.
! This is independent of the legacy solar_reconnection example.
module mod_usr
  use mod_mhd
  implicit none

  double precision :: gravity_code, temperature_ch, temperature_cor
  double precision :: transition_height, transition_width, base_pressure
  double precision :: b_open, guide_ratio, dipole_depth, null_height

contains

  subroutine usr_params_read(files)
    character(len=*), intent(in) :: files(:)
    integer :: n
    namelist /usr_list/ gravity_code, temperature_ch, temperature_cor, &
      transition_height, transition_width, base_pressure, b_open, &
      guide_ratio, dipole_depth, null_height
    do n=1,size(files)
      open(unitpar,file=trim(files(n)),status='old')
      read(unitpar,usr_list,end=111)
111   close(unitpar)
    end do
  end subroutine usr_params_read

  subroutine usr_init()
    use mod_variables
    call usr_params_read(par_files)
    call set_coordinate_system("Cartesian_2.5D")
    unit_length=1.d9
    unit_temperature=1.5d6
    unit_numberdensity=1.d9
    usr_init_one_grid=>initonegrid_usr
    usr_set_B0=>specialset_B0
    usr_gravity=>gravity
    call mhd_activate()
  end subroutine usr_init

  elemental double precision function temperature_profile(y)
    double precision, intent(in) :: y
    temperature_profile=temperature_ch+0.5d0*(temperature_cor-temperature_ch)*&
      (1.d0+dtanh((y-transition_height)/transition_width))
  end function temperature_profile

  elemental double precision function hydrostatic_primitive(y)
    double precision, intent(in) :: y
    double precision :: a,b,u,c
    a=temperature_ch
    b=temperature_cor
    u=2.d0*(y-transition_height)/transition_width
    c=(a-b)/(a*b)
    if (u>=0.d0) then
      hydrostatic_primitive=0.5d0*transition_width*(u/b+c*&
        (dlog(b)+dlog(1.d0+(a/b)*dexp(-u))))
    else
      hydrostatic_primitive=0.5d0*transition_width*(u/a+c*&
        (dlog(a)+dlog(1.d0+(b/a)*dexp(u))))
    endif
  end function hydrostatic_primitive

  elemental double precision function pressure_profile(y)
    double precision, intent(in) :: y
    pressure_profile=base_pressure*dexp(-gravity_code*&
      (hydrostatic_primitive(y)-hydrostatic_primitive(0.d0)))
  end function pressure_profile

  subroutine initonegrid_usr(ixI^L,ixO^L,w,x)
    integer, intent(in) :: ixI^L,ixO^L
    double precision, intent(in) :: x(ixI^S,1:ndim)
    double precision, intent(inout) :: w(ixI^S,1:nw)
    w(ixO^S,rho_)=pressure_profile(x(ixO^S,2))/&
      temperature_profile(x(ixO^S,2))
    w(ixO^S,p_)=pressure_profile(x(ixO^S,2))
    w(ixO^S,mom(:))=zero
    w(ixO^S,mag(:))=zero
    if(mhd_glm) w(ixO^S,psi_)=zero
    call mhd_to_conserved(ixI^L,ixO^L,w,x)
  end subroutine initonegrid_usr

  subroutine specialset_B0(ixI^L,ixO^L,x,wB0)
    integer, intent(in) :: ixI^L,ixO^L
    double precision, intent(in) :: x(ixI^S,1:ndim)
    double precision, intent(inout) :: wB0(ixI^S,1:ndir)
    double precision :: yp(ixI^S),r2(ixI^S),moment
    yp(ixO^S)=x(ixO^S,2)+dipole_depth
    r2(ixO^S)=x(ixO^S,1)**2+yp(ixO^S)**2
    moment=b_open*(null_height+dipole_depth)**2
    wB0(ixO^S,1)=-2.d0*moment*x(ixO^S,1)*yp(ixO^S)/r2(ixO^S)**2
    wB0(ixO^S,2)=b_open-moment*&
      (yp(ixO^S)**2-x(ixO^S,1)**2)/r2(ixO^S)**2
    wB0(ixO^S,3)=guide_ratio*b_open
  end subroutine specialset_B0

  subroutine gravity(ixI^L,ixO^L,wCT,x,gravity_field)
    integer, intent(in) :: ixI^L,ixO^L
    double precision, intent(in) :: x(ixI^S,1:ndim)
    double precision, intent(in) :: wCT(ixI^S,1:nw)
    double precision, intent(out) :: gravity_field(ixI^S,ndim)
    gravity_field(ixO^S,:)=zero
    gravity_field(ixO^S,2)=-gravity_code
  end subroutine gravity

end module mod_usr
