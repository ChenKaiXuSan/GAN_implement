# Codeing log

## 2021年09月04日 22:14:53
- samples/0904_cifar10_oneimage/
  没有成功生成图像。
  - [x]  保存单张图片的功能是正确的，但是序号有问题。已经修改好了。
  输入的imgsize是32，但生成的是64。 不太行。
- 0903_lsun_36_42
  目前为止还没有图片生成出来。不知道是图片太大还是代码出现了问题。
- todo
  尝试生成128的图片
  不加attention的网络结构

## 2021年09月07日 12:38:37 
 
- 0903_lsun_36_42/
  img size 64x64
  生成的的图像没有像素，而且梯度也没有变化。
  不知道是不是因为图像分辨率太大的原因导致的。训练的时间也特别长
- 0904_cifar10_oneimage_noattn_matsumoto
  没有加attn，但是生成的图片还是重复的太多了。

## 2021年09月08日 21:33:30
- 0908_cifar10_noattn_8_16_hatano
  feat改小了之后可以解决生成同一个图片的问题，但是loss抖动变大。
  而且不知道为什么，fid一直都很高，都是100多了。
  还是应该多看论文，目前没有什么好的想法。

## 2021年09月09日 13:32:15
- 0909_cifar10_attn_8_16_matsumoto 
  和上面的参数一样，这次加上attn。
  然后再trainer里面加了随机seed，不知道会不会对结果产生影响。
  log保存在/log/0909.log 里面。

## 2021年09月15日 11:16:43
- 0911_cifar10_noattn_bilinear_add_upsampling_36_42_hatano
  使用了bilinear add upsampling的方法。参数量没怎么变化，但是运算时间估计翻倍了。
  效果看上去也没什么大的变化。
  在100epoch之后出现mode collesp，原因应该是feature太大了图像太简单，以及加入resnet之后模型能力太强的原因。
  收敛速度太快了。  

## 2021年09月17日 11:39:46  
-  0916_cifar10_noattn_bilinear_add_upsampling_8_16_hatano
  缩小feat的size之后训练速度变快了，但是loss的收敛速度变慢了。
  目前看来增加feat的作用还是有的，但是不知道为什么还是会model collapse，现在还是觉得loss function方面太简单导致训练不稳定。
  之后计划 
   - upsampling方面的东西
   - attn方面有没有研究的东西？
   - 解决model collapse问题 

## 2021年09月20日 15:54:57
- 0918_cifar10_attn_bilinear_add_up_aux_matsumoto
  加了aux之后，D正常下降，但是D的loss变得很奇怪，有点不下降的感觉。
  - 之后计划
  - aux的代码好像有问题，在计算loss的地方，还有D里面的结构还有待讨论
    - 目前看来，adv layer用sigmoid, aux layer用softmax计算应该没啥问题，原论文也是这么写的。但是不知道为什么D不收敛。
  - D的中间，应不应该加dropout？
  