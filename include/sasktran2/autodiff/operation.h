#pragma once

#include <sasktran2/internal_common.h>
#include <sasktran2/dual.h>

namespace sasktran2::autodiff {
    enum class AutoDiffMode { Forward, Reverse, Mixed };
    struct Expression;

    using ExprPtr = std::shared_ptr<Expression>;

    struct Expression {
        AutoDiffMode mode;

        Expression(AutoDiffMode m) : mode(m) {}

        virtual ~Expression() = default;

        virtual void forward() = 0;
        virtual void backward() = 0;
    };

    struct UnaryExpression : Expression {
        ExprPtr parent;

        UnaryExpression(ExprPtr p)
            : parent(p), Expression(p ? p->mode : AutoDiffMode::Forward) {}

        void forward() override {
            if (parent) {
                parent->forward();
            }
            forward_impl();
        }

        void backward() override {
            backward_impl();
            if (parent) {
                parent->backward();
            }
        }

      private:
        virtual void forward_impl() = 0;
        virtual void backward_impl() = 0;
    };

    struct NullExpression : Expression {

        NullExpression(AutoDiffMode m) : Expression(m) {}

        virtual void forward() override {
            // Do nothing
        }

        virtual void backward() override {
            // Do nothing
        }
    };

} // namespace sasktran2::autodiff
