def test_root_4(coefficients,root_guess):
  full_root = tf.pow(tf.tile(root_guess,[5]),tf.constant([0,1,2,3,4]))
  result = tf.cumsum(coefficients*full_root)[-1]
  with tf.Session() as sess:
    Res = sess.run(result)
    if Res == 0:
      return True
  return False


def compute_root_4(coef,guess):
  f_org = coef[0] + coef[1]*guess + 2*coef[2]*guess*guess + 3*coef[3]*guess*guess*guess + 4*coef[4]*guess*guess*guess*guess
  f_prime = coef[1] + 2*coef[2]*guess + 3*coef[3]*guess*guess + 4*coef[4]*guess*guess*guess
  f_double_prime = 2*coef[2] + 6*coef[3]*guess + 12*coef[4]*guess*guess
  while test_root_4(coef,guess) == False:
    guess = guess - 2*f_org*f_prime/(2*f_prime*f_prime - f_org*f_double_prime)
  return guess

# a = tf.constant([1,2,3])
# d = tf.cumsum(a)[-1]
# with tf.Session() as sess:
#   D = sess.run(d)
#   print(D)

# a = tf.constant([2,3,4])
# b = tf.constant([2,3,2])
# c = tf.pow(a,b)
# with tf.Session() as sess:
#   C = sess.run(c)
#   print(C)

# a = tf.constant([3])
# a1 = tf.tile(a,[4])
# with tf.Session() as sess:
#   A1 = sess.run(a1)
#   print(A1)
                     
a = tf.constant([0,1,1,1,1])
b = tf.constant([3])
with tf.Session() as sess:
  print(compute_root_4(a,b).eval())
