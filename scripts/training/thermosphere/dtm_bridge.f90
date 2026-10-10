! Independent calling interface; upstream DTM source is compiled unchanged.
! All COMMON real values must be compiled with -fdefault-real-8.
subroutine argus_dtm_init(coefficients) bind(C)
  use iso_c_binding
  implicit none
  real(c_double), intent(in) :: coefficients(96,9)
  real :: tt(96),h(96),he(96),o(96),az2(96),o2(96),az(96),t0(96),tp(96)
  real :: pi,deupi,cdr,sard
  common /pardtm/ tt,h,he,o,az2,o2,az,t0,tp
  common /cons/ pi,deupi,cdr,sard
  tt=coefficients(:,1)
  h=coefficients(:,2)
  he=coefficients(:,3)
  o=coefficients(:,4)
  az2=coefficients(:,5)
  o2=coefficients(:,6)
  az=coefficients(:,7)
  t0=coefficients(:,8)
  tp=coefficients(:,9)
  pi=acos(-1.)
  deupi=2.*pi
  cdr=pi/180.
  sard=0.
end subroutine

subroutine argus_dtm_eval(x,result) bind(C)
  use iso_c_binding
  implicit none
  ! day, altitude km, latitude rad, longitude rad, local solar time rad,
  ! daily F10.7, mean F10.7, delayed Kp, mean Kp
  real(c_double), intent(in) :: x(9)
  real(c_double), intent(out) :: result(3)
  real :: f(2),fm(2),akp(4),d(6),wmm
  f=[x(6),0._c_double]
  fm=[x(7),0._c_double]
  akp=[x(8),0._c_double,x(9),0._c_double]
  call dtm3(x(1),f,fm,akp,x(2),x(5),x(3),x(4),result(2),result(3),result(1),d,wmm)
  result(1)=result(1)*1000._c_double ! g/cm3 -> kg/m3
end subroutine
